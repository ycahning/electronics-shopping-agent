"""Explicit edits, ordinal references, and a bounded offline parser."""

import re
from decimal import Decimal, InvalidOperation
from difflib import get_close_matches
from typing import Literal

from .contracts import HardRequirement, Operation, Patch, Session

ALIASES = {
    "All_Beauty": ("all_beauty", "beauty"),
    "Amazon_Fashion": ("amazon_fashion", "fashion"),
    "Appliances": ("appliances", "appliance"),
    "Arts_Crafts_and_Sewing": ("arts_crafts_and_sewing", "crafts"),
    "Electronics": ("electronics",),
}
DISCOVERY_HELP = "Use a category, a budget, or an explicit search: <keywords>."
BUDGET_QUESTION = "What maximum USD budget should I use?"
CHOICE_CRITERION_QUESTION = "What matters most for this choice, such as price, average rating, or a specific feature?"
EXPLICIT_SEARCH = re.compile(r"\b(?:search|find)\s*:\s*([^;,]+)")
_ASIN = re.compile(
    r"\[([A-Z0-9]{10})\]|\b(?:ASIN\s*[:#]?\s*([A-Z0-9]{10})|((?:B0)[A-Z0-9]{8}))\b",
    re.IGNORECASE,
)
_EXCLUDED_ASIN = re.compile(
    r"\b(?:exclude|omit|avoid|leave out|do not show|don't show)\s+(?:the\s+)?"
    r"(?:(?:ASIN\s*[:#]?\s*)?(?:\[[A-Z0-9]{10}\]|B0[A-Z0-9]{8})|ASIN\s*[:#]?\s*[A-Z0-9]{10})(?!\w)",
    re.IGNORECASE,
)
_EXCLUDED_ORDINAL = re.compile(
    r"\bexclude\s+(?:the\s+)?#?(\d+|first|second|third)\b", re.IGNORECASE
)
_PASTED_CARD = re.compile(r"(?m)^[ \t]*(?:\d+\.|-) [^\n]*\[[A-Za-z0-9]{10}\][^\n]*$")
_CATEGORY_BROWSE = re.compile(
    r"\b(?:(?:any|all|some)\s+|(?:browse|list|find|show(?: me)?|recommend(?: me)?)\s+(?:an? |the )?)"
    r"electronic(?:s|\s+(?:products?|devices?|items?|goods?))?"
    r"(?=\s*(?:[.!?,]*$|(?:for\s+)?(?:under|up to|at most|below|less than|no more than)\b|with (?:a |the )?budget\b))"
)
_CONTEXTUAL = re.compile(
    r"^(?:(?:ok(?:ay)?|and|then|now)[,.!\s]+)*(?:how about|what about|(?:do the )?same for|and the)\b"
)
_FOCUSED = re.compile(r"\b(?:this|that|the same) (?:one|item|product)\b|\bit[.!?\s]*$")
_SHORT_REFERENCE = re.compile(
    r"(?:the\s+)?(?:item\s+|product\s+|option\s+|number\s+|#\s*)?"
    r"(?:first|second|third|fourth|fifth|sixth|\d+(?:st|nd|rd|th)?)"
    r"(?:\s+(?:one|item|product|option))?[.!?]*"
)
_NUMBER = r"(\d{1,9}(?:\.\d{1,2})?)(?!\d|[.,]\d)"
_BUDGET = re.compile(
    r"\b(?:under|below|less than|up to|at most|no more than|max(?:imum)?(?: of)?|budget\s*(?:to|is|of|=|:)?)"
    r"\s*(?:usd\s*|\$\s*)?" + _NUMBER,
    re.IGNORECASE,
)
_POSTPOSED_BUDGET = re.compile(
    r"(?:usd\s*|\$\s*)"
    + _NUMBER
    + r"\s*(?:dollars?\s+)?(?:max(?:imum)?|limit|cap|or less)\b",
    re.IGNORECASE,
)
_NON_MONEY_UNIT = re.compile(
    r"\s*(?:-?\s*(?:(?:usb(?:[- ][ac])?|hdmi|ethernet)\s+)?ports?|"
    r"hours?|minutes?|days?|weeks?|months?|inches|feet|amps?|watts?|volts?|"
    r"mm|cm|kg|lbs?|dB|kHz|GHz|Hz|GB|TB|mAh|ms|W|V|A)\b",
    re.IGNORECASE,
)
_NEGATED_BUDGET = re.compile(
    r"\b(?:not (?:under|below|less than|at most|up to|no more than)|"
    r"(?:do not|don't|never)\s+(?:set|change|lower|raise|remove|clear|drop|adjust|update)\s+"
    r"(?:(?:the|my)\s+)?(?:budget|(?:price |spending )?limit(?:ation)?|(?:price )?cap|price ceiling))\b"
)
_BUDGET_FOLLOWUP = re.compile(
    r"\b(?:change (?:it )?to|(?:bring|get|take|cut|lower|drop|reduce) (?:it|the budget) (?:down )?to)"
    r"\s*(?:usd\s*|\$\s*)?" + _NUMBER,
    re.IGNORECASE,
)
_CAP_EDIT = re.compile(
    r"\b(?:cap (?:it|(?:the )?(?:same |current )?search) at|"
    r"make (?:it|that)(?=\s*(?:usd\s*|\$))|"
    r"(?:set|change|raise|lower|adjust|update|make) (?:the |my )?(?:(?:same |current )?search )?"
    r"(?:budget|(?:price |spending |upper )?(?:cap|limit(?:ation)?|ceiling)) (?:to|at|exactly)|"
    r"(?:my |the )?(?:price |spending |upper )(?:cap|limit(?:ation)?|ceiling) (?:is|of|at)|"
    r"(?:can(?:not|'t)|won't|will not|do not want to|don't want to) spend (?:any )?(?:more than|over))\s*"
    r"(?:usd\s*|\$\s*)?" + _NUMBER,
    re.IGNORECASE,
)
_BUDGET_REMOVE = re.compile(
    r"\b(?:(?:remove|clear|drop) (?:(?:the|my) )?(?:budget|price (?:cap|limit)|spending limit)\b|"
    r"no (?:budget(?: limit)?|price (?:cap|limit)|spending limit)(?=\s*(?:please\b|[.!?,;]|$)))"
)
_KEEP_BUDGET = re.compile(
    r"\b(?:keep (?:the |my )?(?:(?:current|existing|same) )?budget|no budget changes)\b"
)
_BUDGET_LANGUAGE = re.compile(
    r"\b(?:budget|(?:price|spending|upper) (?:cap|limit(?:ation)?|ceiling)|maximum price)\b"
)
_NON_USD = re.compile(r"[€£¥]|\b(?:eur|gbp|cad|cny|rmb|jpy|hkd)\b", re.IGNORECASE)
_USD_AMOUNT = re.compile(r"(?:usd\s*|\$\s*)" + _NUMBER, re.IGNORECASE)
_CLAUSE_BOUNDARY = re.compile(r"\.(?!\d)|[!?;\n]|,\s|\b(?:but|however|whereas)\b")
_HISTORICAL_PRICE = re.compile(
    r"\b(?:saw|bought|paid|(?:advertised|listed|sold|offered) it|was priced|was listed|used to cost|"
    r"last (?:week|month|year)|yesterday|previously)\b"
)
_UNCERTAIN_PRICE = re.compile(r"\b(?:if|maybe|perhaps|possibly|might)\b")
_FOLLOWING_BUDGET_ALTERNATIVE = re.compile(
    r"\s*(?:(?:or|to)\b|[-–—])\s*(?:usd\s*|\$\s*)?" + _NUMBER
)
_SEMANTIC_BUDGET = re.compile(
    r"\b(?:the most i (?:can|will|am willing to) (?:pay|spend|afford)|"
    r"the most i am comfortable spending|all i (?:can|will) (?:pay|spend|afford)|"
    r"(?:my|the) (?:absolute )?(?:maximum|price ceiling)|(?:do not|don't) want to go higher)\b"
)
_UNSUPPORTED = {
    "live stock": ("live stock", "in stock"),
    "live price": ("live price", "current price"),
    "variant fit or specification": ("compatible", "compatibility", "exact size"),
}
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}
_ORDINAL_REF = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth)\b|(?<![\w$.,])([1-9]\d?)(?:st|nd|rd|th)?(?!\w|[.,]\d)"
)
_REVIEWS = re.compile(
    r"\b(?:reviews?|comments?|feedback|evidence)\b|what do (?:buyers|customers|reviewers|shoppers|owners|people) say"
)
_PRICE_PRIORITY = re.compile(
    r"\b(?:cheapest|cheaper|(?:least|less) expensive|more affordable|"
    r"(?:lowest|lower) (?:catalog |listed )?(?:price|cost)|"
    r"(?:price|cost) (?:comes first|matters most|is (?:my |the )?priority)|"
    r"(?:my |the )?(?:main )?priority is (?:price|cost)|"
    r"(?:pay|spend) (?:the )?least)\b"
)
_RATING_PRIORITY = re.compile(
    r"\b(?:highest[- ]rated|best[- ]rated|(?:higher|highest|better|best) "
    r"(?:(?:average|customer|star) )*ratings?|(?:by|based on)\s+(?:the\s+)?"
    r"(?:(?:average|customer|star) )*ratings?|ratings? (?:matters? most|comes? first))\b"
)
_PRICE_NEGATED = re.compile(
    r"\b(?:not|never|don't|do not)\b[^.!?;]{0,42}$|"
    r"\b(?:rather than|instead of|regardless of)\s+(?:the\s+)?$"
)
_HARD_CUE = re.compile(
    r"\b(?:must(?:\s+not)?|need(?:s)? to|has to|have to|required to|only if)\s+"
    r"(?P<value>.+?)(?=,|;|[.!?](?:\s|$)|\s+(?:under|within|at most)\s+\$|$)",
    re.IGNORECASE,
)
_PORT_WORDS = dict(
    zip(
        [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
        ],
        range(1, 13),
        strict=True,
    )
)
_PORTS = re.compile(
    r"\b(?:(?P<relation>at least|at most|minimum(?: of)?|maximum(?: of)?|no fewer than|no more than|exactly)\s+)?"
    r"(?P<number>[1-9]\d?|"
    + "|".join(_PORT_WORDS)
    + r")(?:(?:[- ](?P<port_kind>usb(?:[- ][ac])?|ethernet|hdmi))?\s*[- ]\s*ports?)\b",
    re.IGNORECASE,
)
_USB_VERSION = re.compile(r"\busb\s*([1-4](?:\.[0-2])?)\b", re.IGNORECASE)
_EXCLUDE = re.compile(
    r"\b(?:no|exclude|avoid|without|don't (?:want|show)|do not (?:want|show))\s+"
    r"(?P<target>[^.!?;]{1,100}?)(?=,|;|[.!?](?:\s|$)|\s+(?:under|within)\s+\$|$)",
    re.IGNORECASE,
)
_NEGATED_ATTRIBUTE_PREFIX = re.compile(
    r"\b(?:no|not|exclude|avoid|without|don't (?:want|show)|do not (?:want|show)|"
    r"must not (?:support|use))\s+(?:(?:any|only|the|models with|products with)\s+)*$",
    re.IGNORECASE,
)
_COMPATIBILITY = re.compile(
    r"\b(?:compatib(?:le|ility)\s+with|(?:work|works|working|fit|fits|connect|pair)\s+with)\s+"
    r"(?P<target>.+?)(?=,|;|[.!?](?:\s|$)|\s+(?:under|within|at most)\s+\$|$)",
    re.IGNORECASE,
)
_MODEL_TOKEN = re.compile(r"\b[a-z]{2,}[- ‐‑]\d{2,}[a-z0-9-]*\b", re.IGNORECASE)
_NAMED_TARGET = re.compile(
    r"\b(?:what do (?:buyers|customers|reviewers) say|(?:customer )?reviews?|comments|feedback)"
    r"\s+(?:about|for|of|on)\s+(?:the\s+)?(.+?)[?!.]*\s*$",
    re.IGNORECASE,
)
_LIST_SCOPE = re.compile(
    r"\b(?:these(?:\s+(?:products|items|options|results))?|both(?:\s+of)?\s+(?:them|products|items)|"
    r"all(?:\s+of)?\s+(?:them|the (?:listed|displayed) (?:products|items|options)))\b",
    re.IGNORECASE,
)
_REFERENCE_LIST = re.compile(
    r"(?:\b(?:items?|options?|products?|numbers?|compare|between|for|of|about|vs)\s+(?:no\.?\s+)?|#\s*)"
    r"(?P<numbers>[1-9]\d?(?:st|nd|rd|th)?"
    r"(?:(?:\s*,\s*|\s+(?:and|or|to|vs\.?)\s+)#?\s*[1-9]\d?(?:st|nd|rd|th)?)*)"
    r"(?![\w.,]\d|\w)",
    re.IGNORECASE,
)
_UNQUALIFIED_NUMBER = re.compile(r"(?<![\w$.,])\d+(?:st|nd|rd|th)?(?!\w|[.,]\d)")
HistoryUsage = Literal["catalog_only", "history_preferred", "history_only"]
_HISTORY_SOURCE = (
    r"(?:(?:my|the|selected|this user's|the selected user's|user)\s+)*"
    r"(?:(?:shopping|purchase|interaction|dataset)\s+)?(?:history|historical candidates|profile)"
)
_HISTORY_USE = re.compile(
    r"\b(?:use|consult|check|reference|consider|prefer|look at|include|use results from|"
    r"draw from|draw on|based on|start (?:with|from)|from|using|combine (?:the )?catalog with)\s+"
    + _HISTORY_SOURCE
    + r"\b|\b(?:personalize (?:it|the (?:search|results|recommendations))|"
    r"what (?:might|would) i like next)\b",
    re.IGNORECASE,
)
_HISTORY_DISABLED = re.compile(
    r"\b(?:(?:(?:do not|don't|never) (?:use|consult|check|include|reference)|"
    r"without(?: using)?|ignore|skip|exclude|independent of)\s+"
    + _HISTORY_SOURCE
    + r"\b|(?:do not|don't|never) personalize\b|generic search only\b|"
    r"(?:the )?catalog only\b|(?:use|search) only (?:the )?catalog\b)",
    re.IGNORECASE,
)
_HISTORY_EXCLUSIVE = re.compile(
    r"\b(?:(?:only|solely|exclusively) (?:use |show |recommend |choose )?(?:(?:items|candidates|products|matches) )?"
    r"(?:from |based on |using )?|(?:use|show|recommend|choose) (?:only|solely|exclusively) "
    r"(?:(?:items|candidates|products|matches) )?(?:from |based on |using )?)"
    + _HISTORY_SOURCE
    + r"\b|\b"
    + _HISTORY_SOURCE
    + r"\s+(?:only(?!\s+if\b)|as (?:the )?only source|as (?:the )?sole source)\b",
    re.IGNORECASE,
)
_ASSISTANT_HELP = re.compile(
    r"\b(?:who are you|introduce yourself|what can you (?:do|help with)|"
    r"what are your (?:capabilities|features)|(?:how (?:do|can|does)|"
    r"(?:explain|tell me|describe) how)"
    r"\s+(?:you|the app|this app|this assistant|the assistant|this project)\b)",
    re.IGNORECASE,
)
_SHOPPING_REQUEST = re.compile(
    r"\b(?:find|shop for|search for|look for|show me|recommend|browse)"
    r"\s+(?!(?:how|what|whether|why|ways|about)\b)\w|"
    r"\bi (?:need|want)\s+(?!(?:to (?:learn|know|understand)|you to)\b)\w",
    re.IGNORECASE,
)
_SHOPPING_CONJUNCTION = re.compile(
    r"\b(?:and|then|also)\s+(?:find|show me|recommend|shop for|search for|compare)\b",
    re.IGNORECASE,
)
_REVIEW_REQUEST = re.compile(
    r"\b(?:any reviews?|(?:reviews?|feedback|comments)\s+(?:about|for|of|on)|"
    r"what do (?:buyers|customers|reviewers) say|(?:read|pull|fetch|show(?: me)?)"
    r"\s+(?:the |some )?reviews?)\b",
    re.IGNORECASE,
)
_ITEM_DETAIL_REQUEST = re.compile(
    r"\b(?:price|cost|rating|stock|availability|specifications?|compatibility|how much|"
    r"which (?:one|item|product)|does (?:it|this|that) (?:fit|support|work))\b",
    re.IGNORECASE,
)


def extract_asins(text: str) -> list[str]:
    """Read only ASINs explicitly written by the shopper."""
    return list(
        dict.fromkeys(
            next(part for part in match.groups() if part).upper()
            for match in _ASIN.finditer(text)
        )
    )


def without_pasted_cards(text: str) -> str:
    """Keep pasted product titles and prices out of requirement parsing."""
    return _PASTED_CARD.sub("", text).strip()


def shopper_references(text: str) -> tuple[list[str], str]:
    """Exclude negated product mentions before binding lookup references."""
    positive = _EXCLUDED_ORDINAL.sub("", _EXCLUDED_ASIN.sub("", text))
    asins = extract_asins(positive)
    return asins, without_pasted_cards(positive) if asins else positive


def wants_category_browse(text: str) -> bool:
    s = text.lower().strip()
    return not EXPLICIT_SEARCH.search(s) and bool(
        _CATEGORY_BROWSE.search(s)
        or re.search(r"\bbrowse (?:the )?(?:whole category|entire catalog)\b", s)
    )


def _explicit_followup(s: str) -> str | None:
    if re.search(
        r"\bcompare\b|difference between|first two differ|which (?:one )?is cheaper|"
        r"\bwhich\b[^.?\n]{0,80}\b(?:has|had|is)\b[^.?\n]{0,30}\brat(?:e|ed|ing|ings)\b|"
        r"\bwhich (?:one|of)\b[^.?\n]{0,80}\b(?:cheaper|choose|buy|pick|recommend)\b",
        s,
    ):
        return "compare"
    if asks_for_reviews(s):
        return "evidence"
    return None


def asks_for_reviews(text: str) -> bool:
    return bool(_REVIEWS.search(text.casefold()))


def _positive_priority(text: str, pattern: re.Pattern) -> bool:
    text = text.casefold()
    return any(
        not _PRICE_NEGATED.search(text[: match.start()].rsplit(",", 1)[-1])
        and not re.match(r"\s+(?:does not|doesn't|is not|isn't)\b", text[match.end() :])
        for match in pattern.finditer(text)
    )


def prioritizes_price(text: str) -> bool:
    return _positive_priority(text, _PRICE_PRIORITY)


def prioritizes_rating(text: str) -> bool:
    return _positive_priority(text, _RATING_PRIORITY)


def grounded_comparison_aspect(text: str, query: str) -> bool:
    """Require a stated comparison criterion before looking up its review excerpts."""
    terms = set(re.findall(r"[a-z0-9]+", query.casefold()))
    vague = {
        "a",
        "about",
        "and",
        "based",
        "best",
        "better",
        "buy",
        "buyer",
        "choice",
        "compare",
        "comparison",
        "customer",
        "first",
        "for",
        "good",
        "item",
        "items",
        "one",
        "option",
        "overall",
        "pick",
        "product",
        "products",
        "purchase",
        "quality",
        "recommend",
        "recommendation",
        "review",
        "reviews",
        "second",
        "should",
        "the",
        "to",
        "two",
        "what",
        "which",
    }
    meaningful = terms - vague
    return bool(meaningful) and meaningful <= set(
        re.findall(r"[a-z0-9]+", text.casefold())
    )


def _price_clause(s: str, start: int, end: int) -> str:
    left = 0
    for boundary in _CLAUSE_BOUNDARY.finditer(s):
        if boundary.end() <= start:
            left = boundary.end()
        elif boundary.start() >= end:
            return s[left : boundary.start()]
    return s[left:]


def _budget_matches(s: str, *, followup: bool = False) -> list[re.Match]:
    patterns = [_POSTPOSED_BUDGET, _CAP_EDIT, _BUDGET]
    if followup:
        patterns.append(_BUDGET_FOLLOWUP)
    matches = {}
    for pattern in patterns:
        for match in pattern.finditer(s):
            if not re.search(
                r"\$|\busd\b", match[0], re.IGNORECASE
            ) and _NON_MONEY_UNIT.match(s[match.end() :]):
                continue
            if _HISTORICAL_PRICE.search(_price_clause(s, *match.span())):
                continue
            matches.setdefault(match.span(1), match)
    return list(matches.values())


def _budget_match(s: str) -> re.Match | None:
    return next(iter(_budget_matches(s)), None)


def _ambiguous_budget_amount(s: str, match: re.Match) -> bool:
    clause = _price_clause(s, *match.span())
    return bool(
        _UNCERTAIN_PRICE.search(clause)
        or _FOLLOWING_BUDGET_ALTERNATIVE.match(s[match.end(1) :])
        or len({Decimal(amount[1]) for amount in _USD_AMOUNT.finditer(clause)}) > 1
    )


def grounded_budget_proposal(
    text: str, value: str, source_text: str
) -> Operation | None:
    """Validate the model's proposed ceiling against one explicit, unambiguous USD span."""
    source_text = source_text.strip()
    if (
        not source_text
        or source_text not in text
        or not re.fullmatch(_NUMBER, str(value))
    ):
        return None
    raw, source = text.casefold(), source_text.casefold()
    if (
        _NEGATED_BUDGET.search(raw)
        or _NON_USD.search(raw)
        or _BUDGET_REMOVE.search(raw)
    ):
        return None
    explicit = _budget_matches(raw, followup=True)
    if explicit and (
        {Decimal(match[1]) for match in explicit} != {Decimal(value)}
        or any(_ambiguous_budget_amount(raw, match) for match in explicit)
    ):
        return None
    # Additional wording has to express a ceiling and mark the cited amount as money.
    # A product's observed price or a bare amount is never authorization to change state.
    evidence = _budget_matches(source, followup=True)
    if (
        not evidence
        and _SEMANTIC_BUDGET.search(source)
        and not _HISTORICAL_PRICE.search(source)
    ):
        evidence = list(_USD_AMOUNT.finditer(source))
    if (
        not evidence
        or {Decimal(match[1]) for match in evidence} != {Decimal(value)}
        or any(_ambiguous_budget_amount(source, match) for match in evidence)
    ):
        return None
    for mention in re.finditer(re.escape(source), raw):
        clause = _price_clause(raw, *mention.span())
        if _HISTORICAL_PRICE.search(clause) or _UNCERTAIN_PRICE.search(clause):
            continue
        # On the semantic fallback, a second amount in the source utterance must
        # have a separately established context before accepting either value.
        if (
            not explicit
            and len({Decimal(match[1]) for match in _USD_AMOUNT.finditer(raw)}) != 1
        ):
            continue
        return Operation(field="max_price_usd", op="set", value=str(value))
    return None


def catalog_keywords(query: str) -> str:
    """Keep structured filter clauses out of catalog keyword matching."""
    limit = _budget_match(query.casefold())
    if limit:
        query = query[: limit.start()].rstrip(" ,;")
    return re.split(
        r"\s+(?:compatible with|that (?:must|works?)|(?:with )?(?:at least|at most)"
        r"\s+\w+\s+(?:usb[- ]?[ac]?\s+)?ports?|(?:with )?(?:no|without)\s+usb\b)",
        query,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()


def structured_requirements(
    text: str, extras: list[str] | None = None
) -> list[HardRequirement]:
    """Preserve explicitly stated constraints even when keywords omit them."""
    requirements: list[HardRequirement] = []
    for port in _PORTS.finditer(text):
        port_type = (port["port_kind"] or "").casefold()
        negative = _NEGATED_ATTRIBUTE_PREFIX.search(text[: port.start()])
        if (port_type and port_type != "usb") or negative:
            clause = text[negative.start() : port.end()] if negative else port[0]
            requirements.append(
                HardRequirement(kind="property", value=clause.casefold(), text=clause)
            )
            continue
        token = port["number"].casefold()
        number = int(token) if token.isdecimal() else _PORT_WORDS[token]
        relation = (port["relation"] or "exactly").casefold()
        if relation.startswith(("at least", "minimum", "no fewer than")):
            kind = "min_ports"
        elif relation.startswith(("at most", "maximum", "no more than")):
            kind = "max_ports"
        else:
            kind = "exact_ports"
        requirements.append(HardRequirement(kind=kind, value=str(number), text=port[0]))
    for version in _USB_VERSION.finditer(text):
        prefix = _NEGATED_ATTRIBUTE_PREFIX.search(text[: version.start()])
        kind = "excluded_usb_version" if prefix else "required_usb_version"
        clause = text[prefix.start() : version.end()] if prefix else version[0]
        requirements.append(HardRequirement(kind=kind, value=version[1], text=clause))
    for fit in _COMPATIBILITY.finditer(text):
        target = fit["target"].strip()[:200]
        requirements.append(
            HardRequirement(kind="compatibility", value=target, text=fit[0][:200])
        )
    possibilities = [match[0] for match in _HARD_CUE.finditer(text)]
    possibilities.extend(match[0] for match in _EXCLUDE.finditer(text))
    possibilities.extend(
        extra for extra in extras or [] if extra.casefold() in text.casefold()
    )
    for clause in possibilities:
        clause = clause.strip()[:200]
        value = clause.casefold()
        if (
            not clause
            or value == "no budget"
            or _budget_match(value)
            or re.match(r"exclude\s+#?(?:\d+|first|second|third)\b", value)
            or any(
                value in req.text.casefold() or req.text.casefold() in value
                for req in requirements
            )
        ):
            continue
        requirements.append(HardRequirement(kind="property", value=value, text=clause))
    unique = {(req.kind, req.value.casefold()): req for req in requirements}
    return list(unique.values())[:12]


def history_request(text: str, patch: Patch) -> bool:
    """Recognize an explicit request for the model's history ranking on its own."""
    if (
        patch.intent in {"answer", "compare", "evidence"}
        or history_usage(text) == "catalog_only"
    ):
        return False
    if patch.requirements or any(
        op.field in {"max_price_usd", "excluded_id", "unsupported"}
        for op in patch.operations
    ):
        return False
    generic = {
        "electronic",
        "electronics",
        "product",
        "products",
        "item",
        "items",
        "recommendation",
        "recommendations",
        "history",
        "next",
        "personalized",
        "top",
        "shopping",
        "purchase",
        "interaction",
        "recommend",
        "user",
        "dataset",
    }
    literal_query = set(re.findall(r"[a-z]+", patch.query.casefold()))
    literal_request = set(re.findall(r"[a-z]+", text.casefold()))
    return not (literal_query & literal_request) - generic


def history_usage(text: str) -> HistoryUsage:
    """Return the history scope explicitly requested in this user message."""
    if _HISTORY_DISABLED.search(text):
        return "catalog_only"
    if _HISTORY_EXCLUSIVE.search(text) or re.search(
        r"\bnext[- ]item\b", text, re.IGNORECASE
    ):
        return "history_only"
    if _HISTORY_USE.search(text):
        return "history_preferred"
    return "catalog_only"


def history_count(text: str) -> int | None:
    match = re.search(
        r"\b(?:top|next|show(?: me)?|give(?: me)?|recommend)\s+(\d{1,3})\b",
        text,
        re.IGNORECASE,
    )
    return int(match[1]) if match else None


def _ordinal_references(s: str) -> list[int]:
    # A price is not an item number. Keep the original order for comparisons.
    s = re.sub(r"\$\s*\d+(?:[.,]\d+)*|\busd\s+\d+(?:[.,]\d+)*", "", s)
    s = _PORTS.sub("", s)
    s = _USB_VERSION.sub("", s)
    refs = [1, 2] if "first two" in s else [1, 2, 3] if "first three" in s else []
    for word, number in _ORDINAL_REF.findall(s):
        ordinal = ORDINALS[word] if word else int(number)
        if ordinal not in refs:
            refs.append(ordinal)
    return refs


def _grounded_ordinals(s: str) -> list[int]:
    named_labels = [
        _PORT_WORDS[match]
        for match in re.findall(
            r"\b(?:item|option|product|number)\s+(?:no\.?\s+)?("
            + "|".join(_PORT_WORDS)
            + r")\b",
            s,
        )
    ]
    if _SHORT_REFERENCE.fullmatch(s):
        return _ordinal_references(s)
    s = _PORTS.sub("", _USB_VERSION.sub("", s))
    spans = [match.span("numbers") for match in _REFERENCE_LIST.finditer(s)]
    restricted = _UNQUALIFIED_NUMBER.sub(
        lambda match: (
            match[0]
            if any(start <= match.start() < end for start, end in spans)
            else ""
        ),
        s,
    )
    return list(
        dict.fromkeys([*_ordinal_references(restricted), *named_labels])
    ) or _fuzzy_ordinal(s)


def _fuzzy_ordinal(s: str) -> list[int]:
    """Recognize a mistyped ordinal only inside an explicit reference phrase."""
    contextual = _CONTEXTUAL.search(s)
    phrase = s[contextual.end() :].strip() if contextual else s
    match = re.fullmatch(
        r"(?:the\s+)?([a-z]{4,7})\s+(?:one|item|option|product)[.?!]*",
        phrase,
    )
    if not match:
        return []
    candidates = get_close_matches(match[1], ORDINALS, n=2, cutoff=0.8)
    return [ORDINALS[candidates[0]]] if len(candidates) == 1 else []


def _is_reference_reply(s: str) -> bool:
    return bool(
        _SHORT_REFERENCE.fullmatch(s)
        or _fuzzy_ordinal(s)
        or re.fullmatch(
            r"(?:item|option|product|number)\s+(?:no\.?\s+)?(?:"
            + "|".join(_PORT_WORDS)
            + r")[.!?]*",
            s,
        )
    )


def grounded_product_names(text: str, names: list[str]) -> list[str]:
    """Accept explicit product mentions, never names introduced by the model."""
    s = text.casefold()
    deterministic = _NAMED_TARGET.search(text)
    if deterministic and _MODEL_TOKEN.search(deterministic[1]):
        names = [deterministic[1]]
    grounded = (
        name.strip()
        for name in names
        if 2 <= len(name.strip()) <= 160
        and name.strip().casefold() in s
        and not _is_reference_reply(name.strip().casefold())
        and not _FOCUSED.search(name.strip().casefold())
        and not _LIST_SCOPE.search(name.strip().casefold())
    )
    return list(dict.fromkeys(grounded))[:3]


def grounded_shopping_intent(text: str, offline: Patch, names: list[str]) -> str | None:
    """Keep explicit shopping requests on the verified path if the model proposes a direct reply."""
    s = text.casefold().strip()
    addresses_items = bool(
        extract_asins(text)
        or _grounded_ordinals(s)
        or (_FOCUSED.search(s) and _ITEM_DETAIL_REQUEST.search(s))
        or _LIST_SCOPE.search(s)
        or grounded_product_names(text, names)
    )
    if addresses_items and offline.intent in {"compare", "evidence"}:
        return offline.intent
    if _ASSISTANT_HELP.search(s) and not _SHOPPING_CONJUNCTION.search(s):
        return None
    if _REVIEW_REQUEST.search(s):
        return "evidence"
    if history_request(text, offline):
        return "recommend"
    if addresses_items:
        return "compare"
    if (
        EXPLICIT_SEARCH.search(s)
        or wants_category_browse(text)
        or _SHOPPING_REQUEST.search(s)
        or any(
            op.field in {"max_price_usd", "excluded_id"} for op in offline.operations
        )
    ):
        return "discover"
    return None


def contextualize_followup(
    text: str, patch: Patch, state: Session, asins: list[str]
) -> Patch:
    """Carry a narrowly scoped follow-up without changing a new shopping request."""
    s = text.lower().strip()
    explicit = _explicit_followup(s)
    previous = state.followup
    contextual = _CONTEXTUAL.search(s)
    reference_text = s[contextual.end() :].strip() if contextual else s
    short_reference = _is_reference_reply(reference_text)
    literal_refs = (
        _ordinal_references(reference_text) or _fuzzy_ordinal(reference_text)
        if short_reference
        else []
    )
    references = literal_refs or patch.references
    focused = bool(_FOCUSED.search(s))
    inherited = bool(
        previous
        and not patch.operations
        and (
            short_reference
            or (contextual and (focused or (references and patch.intent != "discover")))
        )
    )
    if patch.intent in {"answer", "recommend"} or (not asins and not inherited):
        return patch
    explicit_id_request = bool(
        asins
        and offline_parse(text, state, conversational=True).clarification
        in (None, DISCOVERY_HELP)
    )
    if (
        patch.clarification
        and patch.clarification != DISCOVERY_HELP
        and not (explicit_id_request or (inherited and literal_refs))
    ):
        return patch
    if len(asins) > 3 or len(references) > 3:
        return Patch(clarification="Choose up to three products for this request.")
    intent = explicit or (
        previous.intent if previous and (inherited or asins) else None
    )
    if intent is None:
        intent = patch.intent if patch.intent != "discover" else "compare"
    query = patch.query or (
        previous.query if inherited and not explicit and previous else ""
    )
    priority = patch.priority
    if inherited and not explicit and previous:
        priority = priority or previous.priority
    return patch.model_copy(
        update={
            "intent": intent,
            "references": references,
            "query": query,
            "product_names": [],
            "priority": priority if intent == "compare" else None,
            "clarification": None,
        }
    )


def resume_reference(
    text: str, patch: Patch, state: Session, asins: list[str]
) -> Patch:
    pending = state.pending_reference
    if (
        pending is None
        or pending.snapshot != state.snapshot
        or pending.listed_ids != state.displayed_ids
        or (
            not _is_reference_reply(text.casefold().strip())
            and not (
                asins
                and not _explicit_followup(text.casefold())
                and not patch.query
                and not patch.requirements
            )
        )
    ):
        return patch
    return Patch(
        intent=pending.intent,
        query=pending.query,
        priority=pending.priority,
        references=_grounded_ordinals(text.casefold()),
    )


def resume_comparison(
    text: str, patch: Patch, state: Session
) -> tuple[Patch, list[str]]:
    """Apply a shopper's chosen criterion to the same selected items."""
    followup = state.followup
    if not (
        state.pending_question
        and (
            state.pending_question == CHOICE_CRITERION_QUESTION
            or re.search(
                r"\b(?:matter\w*|prefer\w*|priorit\w*|criterion|criteria|choose|important)\b",
                state.pending_question,
                re.IGNORECASE,
            )
        )
        and followup
        and followup.intent == "compare"
        and followup.ids
    ):
        return patch, []
    preference = re.sub(
        r"^(?:(?:i care (?:most )?about|my priority is|focus on|based on|the)\s+)",
        "",
        text.casefold().strip(" .!?"),
    )
    preference = re.sub(r"\s+(?:matters? most|comes? first)$", "", preference)
    if re.fullmatch(
        r"(?:catalog )?(?:price|cost)|affordability|cheapest one", preference
    ):
        resumed = Patch(intent="compare", priority="price")
    elif re.fullmatch(r"(?:(?:average|customer|star) )*(?:rating|ratings)", preference):
        resumed = Patch(intent="compare", priority="rating")
    elif patch.intent != "answer" and re.fullmatch(
        r"(?!(?:hi|hello|hey|thanks?|no|what|which|how|find|show|i|new|start|search|browse|retry|reset|give|recommend|tell)\b)"
        r"[a-z][a-z0-9-]*(?: [a-z0-9-]+){0,3}",
        preference,
    ):
        resumed = Patch(intent="compare", priority="review_aspect", query=preference)
    else:
        return patch, []
    return resumed, followup.ids


def offline_parse(text: str, state: Session, *, conversational: bool = False) -> Patch:
    """Recognize explicit phrases while the model handles open-ended requests."""
    s = text.lower().strip()
    ops: list[Operation] = []
    if _NEGATED_BUDGET.search(s):
        return Patch(clarification="Please confirm the exact requirement to change.")
    if s in {"reset", "clear all"}:
        return Patch(operations=[Operation(field="category", op="clear")])
    found = [
        category for category, terms in ALIASES.items() if any(t in s for t in terms)
    ]
    if (
        "Electronics" not in found
        and _CATEGORY_BROWSE.search(s)
        and wants_category_browse(s)
    ):
        found.append("Electronics")
    if len(found) > 1:
        return Patch(clarification="Which one product category should I use?")
    if found:
        ops.append(Operation(field="category", op="set", value=found[0]))
    if re.search(r"budget[^\n]{0,15}(?:-\s*\d|nan|inf(?:inity)?)", s):
        return Patch(clarification="Enter a finite, nonnegative USD budget.")
    if _NON_USD.search(s):
        return Patch(clarification="This application can check USD budgets only.")
    intent = _explicit_followup(s) or (
        "evidence" if not conversational and "noise" in s else "discover"
    )

    matches = _budget_matches(s, followup=state.max_price_usd is not None)
    if (
        not matches
        and state.pending_question
        and "USD budget" in state.pending_question
    ):
        amount = re.fullmatch(
            r"(?:usd\s*|\$\s*)?" + _NUMBER + r"(?:\s*(?:usd|dollars?))?[.!?]?", s
        )
        if amount:
            matches = [amount]
    values = {Decimal(match[1]) for match in matches}
    remove_budget = _BUDGET_REMOVE.search(s)
    if (
        len(values) > 1
        or (remove_budget and values)
        or any(_ambiguous_budget_amount(s, match) for match in matches)
    ):
        return Patch(clarification=BUDGET_QUESTION)
    if remove_budget:
        ops.append(Operation(field="max_price_usd", op="remove"))
    elif matches:
        ops.append(Operation(field="max_price_usd", op="set", value=matches[0][1]))
    else:
        # A historical price can be useful context without changing the active filter.
        mentions_limit = (
            _BUDGET_LANGUAGE.search(s) or _SEMANTIC_BUDGET.search(s)
        ) and not _HISTORICAL_PRICE.search(s)
        unclear_shopping_price = (
            intent == "discover"
            and "$" in s
            and _SHOPPING_REQUEST.search(s)
            and not _HISTORICAL_PRICE.search(s)
        )
        if (intent == "discover" and re.search(r"\bcheaper\b", s)) or (
            (mentions_limit or unclear_shopping_price) and not _KEEP_BUDGET.search(s)
        ):
            return Patch(clarification=BUDGET_QUESTION)
    if "clear exclusions" in s:
        ops.append(Operation(field="excluded_id", op="clear"))
    exclude = _EXCLUDED_ORDINAL.search(s)
    if exclude:
        token = exclude[1]
        n = ORDINALS.get(token, int(token) if token.isdecimal() else 0)
        if not 1 <= n <= len(state.displayed_ids):
            return Patch(clarification="Which displayed product should I exclude?")
        ops.append(
            Operation(field="excluded_id", op="add", value=state.displayed_ids[n - 1])
        )
    structured = (
        conversational or state.category == "Electronics" or "Electronics" in found
    )
    if (
        re.search(r"\bexclude\b", s)
        and not exclude
        and "clear exclusions" not in s
        and not structured
    ):
        ops.append(
            Operation(field="unsupported", op="add", value="unverified exclusion")
        )
    if re.search(r"clear unsupported|remove unsupported", s):
        ops.append(Operation(field="unsupported", op="clear"))
    requirements = (
        structured_requirements(text)
        if structured and (intent == "discover" or _HARD_CUE.search(text))
        else []
    )
    for condition, words in _UNSUPPORTED.items():
        if any(word in s for word in words) and not (
            structured
            and condition == "variant fit or specification"
            and any(req.kind == "compatibility" for req in requirements)
        ):
            ops.append(Operation(field="unsupported", op="add", value=condition))
    mandatory = re.search(r"\bmust (?:be|have|support)\s+(.+)", s)
    if mandatory and not structured and not _budget_match(mandatory.group(0)):
        condition = "mandatory property not validated by the catalog"
        ops.append(Operation(field="unsupported", op="add", value=condition))
    refs = _ordinal_references(s) if intent != "discover" else []
    if len(refs) > 3:
        return Patch(clarification="Choose up to three products for this request.")
    query = next(
        (
            term
            for term in (
                "kettle",
                "toaster",
                "blender",
                "grinder",
                "vacuum",
                "lipstick",
                "moisturizer",
                "jeans",
                "shirt",
                "yarn",
                "scissors",
                "noise",
            )
            if term in s
        ),
        "",
    )
    if intent == "discover":
        explicit_query = EXPLICIT_SEARCH.search(s)
        if explicit_query:
            query = catalog_keywords(explicit_query[1])[:160]
        elif (
            not query
            and not ops
            and not requirements
            and s not in {"search", "find", "retry", "list"}
        ):
            return Patch(clarification=DISCOVERY_HELP)
    return Patch(
        operations=ops,
        intent=intent,
        references=refs,
        query=query,
        product_names=grounded_product_names(text, []) if intent != "discover" else [],
        requirements=requirements,
        priority=(
            "rating"
            if intent == "compare" and prioritizes_rating(s)
            else "price"
            if intent == "compare" and prioritizes_price(s)
            else None
        ),
    )


def apply_patch(
    state: Session, patch: Patch, *, explicit_exclusions: frozenset[str] = frozenset()
) -> Session:
    """Apply explicit operations as one validated replacement, never a partial mutation."""
    original = state.model_dump()
    data = original.copy()
    if patch.clarification:
        return state
    search_changed = patch.clear_query or (
        patch.query and patch.query.casefold() != state.search_query.casefold()
    )
    if (
        state.category == "Electronics"
        and patch.intent == "discover"
        and search_changed
    ):
        data.update(requirements=[], unsupported=[])
    for op in patch.operations:
        field, kind, val = op.field, op.op, op.value
        if field == "category":
            if kind == "clear":
                data.update(
                    category=None,
                    max_price_usd=None,
                    excluded_ids=[],
                    unsupported=[],
                    requirements=[],
                    displayed_ids=[],
                    search_query="",
                    followup=None,
                    pending_reference=None,
                )
            elif kind == "set" and val:
                if data["category"] != val:
                    data["excluded_ids"] = []
                    data["displayed_ids"] = []
                    data["search_query"] = ""
                    data["followup"] = None
                    data["pending_reference"] = None
                    data["requirements"] = []
                data[field] = val
            else:
                raise ValueError("Unsupported category operation.")
        elif field == "max_price_usd":
            if kind in {"remove", "clear"}:
                data[field] = None
            elif kind == "set" and val is not None:
                try:
                    data[field] = Decimal(val)
                except InvalidOperation as error:
                    raise ValueError("Invalid budget.") from error
            else:
                raise ValueError("Unsupported budget operation.")
        else:
            key = "excluded_ids" if field == "excluded_id" else field
            if kind == "clear":
                data[key] = []
                if key == "unsupported":
                    data["requirements"] = []
            elif kind == "add" and val:
                if (
                    field == "excluded_id"
                    and val not in state.displayed_ids
                    and val not in explicit_exclusions
                ):
                    raise ValueError(
                        "Exclusion is not a supplied ASIN or displayed product."
                    )
                data[key] = list(dict.fromkeys([*data[key], val]))
            elif kind == "remove" and val:
                data[key] = [item for item in data[key] if item != val]
            else:
                raise ValueError("Unsupported list operation.")
    if patch.requirements:
        previous = {
            (req["kind"], req["value"].casefold()): req for req in data["requirements"]
        }
        for requirement in patch.requirements:
            key = requirement.kind, requirement.value.casefold()
            previous[key] = requirement.model_dump()
        if len(previous) > 12:
            raise ValueError("Too many active explicit requirements.")
        data["requirements"] = list(previous.values())
    if data != original:
        data["revision"] += 1
    return Session.model_validate(data)


def resolve_references(
    state: Session,
    patch: Patch,
    *,
    text: str = "",
    asins: list[str] | None = None,
    strict: bool = False,
) -> list[str]:
    if patch.intent in {"answer", "discover", "recommend"}:
        return []
    if asins:
        return asins
    s = text.casefold()
    literal_refs = _grounded_ordinals(s) if strict else _ordinal_references(s)
    if _FOCUSED.search(s) and not literal_refs:
        focused_ids = state.followup.ids if state.followup else state.displayed_ids
        if len(focused_ids) == 1:
            return focused_ids
        raise ValueError("Which product do you mean? Use its number or ASIN.")
    if not state.displayed_ids:
        raise ValueError("Discover products first, so I can refer to what you saw.")
    references = literal_refs if strict else patch.references
    if not references and strict and not _LIST_SCOPE.search(s):
        raise ValueError("Which product do you mean? Use its name, number, or ASIN.")
    if not references:
        return state.displayed_ids[:3]
    if any(not 1 <= n <= len(state.displayed_ids) for n in references):
        raise ValueError("A requested product number was not in the displayed list.")
    return list(dict.fromkeys(state.displayed_ids[n - 1] for n in references))
