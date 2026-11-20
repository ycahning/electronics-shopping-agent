"""Protocol tests for the Ollama parse-only provider without an installed LLM."""

import io
import json

import pytest

from shopping_agent import ollama
from shopping_agent.ollama import OllamaProvider
from shopping_agent.provider import ProviderError

TAG = "qwen3:4b-q4_K_M"


class StubResponse(io.BytesIO):
    def __init__(self, payload, status=200):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        super().__init__(data)
        self.status = status


@pytest.fixture
def mock_http(monkeypatch):
    responses = []
    requests = []
    closed = []

    class StubConnection:
        def __init__(self, host, port, timeout):
            self.address = (host, port, timeout)

        def request(self, method, path, *, body, headers):
            requests.append((self.address, method, path, body, headers))

        def getresponse(self):
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        def close(self):
            closed.append(self.address)

    monkeypatch.setattr(ollama, "HTTPConnection", StubConnection)
    return responses, requests, closed


def test_parse_uses_native_chat_api_with_constrained_patch(mock_http):
    expected = {
        "operations": [],
        "intent": "discover",
        "references": [],
        "query": "wireless headphones",
        "clarification": None,
        "clear_query": False,
    }
    responses, requests, closed = mock_http
    responses.extend(
        [
            StubResponse({"models": [{"name": TAG}]}),
            StubResponse(
                {
                    "done": True,
                    "message": {
                        "content": json.dumps(
                            {
                                key: value
                                for key, value in expected.items()
                                if key != "operations"
                            }
                        )
                    },
                    "prompt_eval_count": 480,
                    "eval_count": 54,
                }
            ),
        ]
    )
    provider = OllamaProvider(TAG, base_url="http://localhost:11435")
    provider.check_ready()
    state = {"search_query": "", "available_categories": ["Electronics"]}
    text = "I need wireless headphones"
    assert provider.parse(text, state, 20) == expected
    assert provider.name == f"ollama:{TAG}"
    assert provider.local_intents
    assert provider.last_usage == {"input_tokens": 480, "output_tokens": 54}

    assert requests[0][:3] == (("127.0.0.1", 11435, 3.0), "GET", "/api/tags")
    address, method, path, body, headers = requests[1]
    assert (address, method, path) == (("127.0.0.1", 11435, 20), "POST", "/api/chat")
    assert headers["Content-Type"] == "application/json"
    assert "Authorization" not in headers
    assert closed == [requests[0][0], requests[1][0]]
    payload = json.loads(body)
    assert payload["model"] == TAG
    assert payload["stream"] is False and payload["think"] is False
    assert payload["format"]["additionalProperties"] is False
    assert "clear_query" in payload["format"]["required"]
    assert "operations" not in payload["format"]["properties"]
    assert json.loads(payload["messages"][1]["content"]) == {
        "user_request": text,
        "state": {**state, "category": None, "displayed_count": 0},
    }


def test_greeting_keeps_only_the_conversational_question(mock_http):
    responses, _, _ = mock_http
    reply = "Hi! What would you like to shop for?"
    responses.append(
        StubResponse(
            {
                "done": True,
                "message": {
                    "content": json.dumps(
                        {
                            "intent": "clarify",
                            "references": [],
                            "query": "phantom search",
                            "clarification": reply,
                            "clear_query": True,
                        }
                    )
                },
            }
        )
    )
    patch = OllamaProvider(TAG).parse("Hello there!", {}, 20)
    assert patch["clarification"] == reply
    assert patch["operations"] == [] and patch["query"] == ""
    assert patch["clear_query"] is False


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (StubResponse({"error": "internal URL"}, 404), "OLLAMA_MODEL_MISSING"),
        (StubResponse({"location": "https://example.com"}, 308), "OLLAMA_OUTPUT"),
        (StubResponse({"error": "runner exited"}, 503), "OLLAMA_UNAVAILABLE"),
        (TimeoutError(), "MODEL_TIMEOUT"),
        (ConnectionRefusedError(), "OLLAMA_UNAVAILABLE"),
        (StubResponse({"done": True, "done_reason": "length"}), "MODEL_INCOMPLETE"),
        (StubResponse({"done": True, "message": {"content": "{}"}}), "INVALID_PATCH"),
        (StubResponse(b"["), "OLLAMA_OUTPUT"),
        (StubResponse(b"a" * (ollama.MAX_RESPONSE_BYTES + 1)), "OLLAMA_OUTPUT"),
    ],
)
def test_bad_chat_responses_raise_safe_provider_errors(mock_http, response, code):
    responses, requests, closed = mock_http
    responses.append(response)
    provider = OllamaProvider(TAG)
    with pytest.raises(ProviderError) as error:
        provider.parse("shopping", {}, 20)
    assert error.value.code == code
    assert "internal URL" not in str(error.value)
    assert len(requests) == len(closed) == 1


def test_preflight_requires_installed_tag_and_rejects_nonlocal_configuration(mock_http):
    responses, requests, _ = mock_http
    responses.append(StubResponse({"models": [{"model": "other:latest"}]}))
    with pytest.raises(ProviderError) as error:
        OllamaProvider(TAG).check_ready()
    assert error.value.code == "OLLAMA_MODEL_MISSING"

    with pytest.raises(ProviderError) as error:
        OllamaProvider(TAG, base_url="https://example.com")
    assert error.value.code == "OLLAMA_ADDRESS"
    with pytest.raises(ProviderError) as error:
        OllamaProvider("gpt-oss:cloud")
    assert error.value.code == "OLLAMA_LOCAL_MODEL"
    assert len(requests) == 1
