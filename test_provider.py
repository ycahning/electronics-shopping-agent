"""Offline tests for the provider boundary; no API key or network is needed."""

import json
import unittest
from types import SimpleNamespace as Record
from unittest.mock import Mock

from shopping_agent.provider import MAX_OUTPUT_TOKENS, ProviderError, ResponsesProvider


def response(*output, status="completed", text="", usage=None):
    return Record(status=status, output=list(output), output_text=text, usage=usage)


def action(name, arguments="{}"):
    return Record(type="function_call", name=name, arguments=arguments)


class ResponsesProviderTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.provider = ResponsesProvider("fixture-model", client=self.client)

    def test_patch_request_uses_closed_structured_output_and_sanitized_usage(self):
        patch = {
            "operations": [{"field": "max_price_usd", "op": "set", "value": "40.00"}],
            "intent": "discover",
            "references": [],
            "query": "",
            "clarification": None,
        }
        self.client.responses.create.return_value = response(
            text=json.dumps(patch),
            usage=Record(
                input_tokens=14,
                output_tokens=9,
                total_tokens=23,
                input_tokens_details=Record(cached_tokens=4),
                private="do not record",
            ),
        )

        self.assertEqual(self.provider.parse("Budget $40", {}, 0.75), patch)

        request = self.client.responses.create.call_args.kwargs
        self.assertFalse(request["store"])
        self.assertEqual(request["timeout"], 0.75)
        self.assertEqual(request["max_output_tokens"], MAX_OUTPUT_TOKENS)
        fmt = request["text"]["format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertTrue(fmt["strict"])
        self.assertFalse(fmt["schema"]["additionalProperties"])
        self.assertEqual(
            self.provider.last_usage,
            {
                "input_tokens": 14,
                "output_tokens": 9,
                "total_tokens": 23,
                "cached_input_tokens": 4,
            },
        )

    def test_decision_uses_one_call_and_omits_nullable_optional_argument(self):
        self.client.responses.create.return_value = response(
            Record(type="reasoning", summary=["not application output"]),
            action("get_product_evidence", '{"ids":["A"],"query":null}'),
        )

        self.assertEqual(
            self.provider.decide({"observed_ids": ["A"]}, 2),
            {
                "action": "get_product_evidence",
                "args": {"ids": ["A"]},
            },
        )
        request = self.client.responses.create.call_args.kwargs
        self.assertFalse(request["parallel_tool_calls"])
        self.assertEqual(request["tool_choice"], "required")
        self.assertNotIn("previous_response_id", request)
        self.assertIsNone(self.provider.last_usage)

    def test_invalid_actions_and_multiple_calls_are_rejected(self):
        scenarios = [
            (response(), "INVALID_ACTION_COUNT"),
            (
                response(action("finish_decision"), action("finish_decision")),
                "INVALID_ACTION_COUNT",
            ),
            (response(action("arbitrary_url")), "UNKNOWN_ACTION"),
            (
                response(action("get_products", '{"ids": ["A"], "unexpected": true}')),
                "INVALID_ACTION_ARGS",
            ),
        ]
        for reply, code in scenarios:
            with self.subTest(code=code):
                self.client.responses.create.return_value = reply
                with self.assertRaises(ProviderError) as raised:
                    self.provider.decide({}, 2)
                self.assertEqual(raised.exception.code, code)

    def test_refusal_incomplete_and_invalid_patch_are_distinct(self):
        scenarios = [
            (
                response(
                    Record(
                        type="message",
                        content=[Record(type="refusal", refusal="private")],
                    )
                ),
                "MODEL_REFUSAL",
            ),
            (response(status="incomplete"), "MODEL_INCOMPLETE"),
            (response(text='{"intent":"discover"}'), "INVALID_PATCH"),
        ]
        for reply, code in scenarios:
            with self.subTest(code=code):
                self.client.responses.create.return_value = reply
                with self.assertRaises(ProviderError) as raised:
                    self.provider.parse("Budget $40", {}, 2)
                self.assertEqual(raised.exception.code, code)
                self.assertNotIn("private", str(raised.exception))

    def test_no_request_is_sent_when_the_full_payload_exceeds_budget(self):
        with self.assertRaises(ProviderError) as raised:
            self.provider.decide({"observations": "a" * 19_500}, 2)
        self.assertEqual(raised.exception.code, "MODEL_CONTEXT_LIMIT")
        self.client.responses.create.assert_not_called()

    def test_timeout_message_is_sanitized_and_only_one_sdk_attempt_is_made(self):
        self.client.responses.create.side_effect = TimeoutError("sensitive raw request")
        with self.assertRaises(ProviderError) as raised:
            self.provider.parse("Budget $40", {}, 2)
        self.assertEqual(raised.exception.code, "MODEL_TIMEOUT")
        self.assertNotIn("sensitive", str(raised.exception))
        self.client.responses.create.assert_called_once()

    def test_locked_sdk_round_trip_uses_only_in_memory_http(self):
        # Core-only installs deliberately omit the live extra. When available, this
        # exercises the real SDK request serialization and typed response parsing.
        try:
            import httpx
            from openai import OpenAI
        except ImportError:
            self.skipTest("live extra not installed")

        requests = []

        def handle(request):
            self.assertEqual(request.url.path, "/v1/responses")
            body = json.loads(request.content)
            requests.append(body)
            if "text" in body:
                output = [
                    {
                        "type": "message",
                        "id": "msg_mock",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "annotations": [],
                                "text": json.dumps(
                                    {
                                        "operations": [],
                                        "intent": "discover",
                                        "references": [],
                                        "query": "kettle",
                                        "clarification": None,
                                    }
                                ),
                            }
                        ],
                    }
                ]
            else:
                output = [
                    {
                        "type": "function_call",
                        "name": "finish_decision",
                        "call_id": "call_mock",
                        "arguments": "{}",
                        "status": "completed",
                    }
                ]
            return httpx.Response(
                200,
                json={
                    "id": "resp_mock",
                    "object": "response",
                    "created_at": 0,
                    "model": "fixture-model",
                    "status": "completed",
                    "output": output,
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                },
            )

        with OpenAI(
            api_key="fake-offline-test-key",
            max_retries=0,
            base_url="https://mock.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as sdk:
            provider = ResponsesProvider("fixture-model", client=sdk)
            self.assertEqual(provider.parse("Find a kettle", {}, 1)["query"], "kettle")
            self.assertEqual(
                provider.decide({}, 1), {"action": "finish_decision", "args": {}}
            )

        self.assertEqual(len(requests), 2)
        self.assertTrue(all(body["store"] is False for body in requests))
