import hmac
import html
import ipaddress
import math
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
ALLOWED_TEMPLATE_KWARGS = frozenset({"open_thinking", "enable_thinking"})

MAX_MESSAGES = 64
MAX_MESSAGE_CHARS = 16_000
MAX_TOTAL_CHARS = 64_000
MAX_TOKENS_MIN = 1
MAX_TOKENS_MAX = 4096
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0
TOP_P_MIN = 0.01
TOP_P_MAX = 1.0
MAX_TOOLS = 16
MAX_MODEL_NAME_LEN = 128

_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_CONTROL_TOKEN_RE = re.compile(
    r"<\|(?:im_start|im_end|endoftext|system|assistant|user|pad|eos|bos)\|>"
    r"|</?s>",
    re.IGNORECASE,
)
_TOOL_TAG_RE = re.compile(r"</?tool_call>", re.IGNORECASE)
_ROLE_OVERRIDE_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts)", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*override", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
    re.compile(r"<\|im_start\|>\s*system", re.IGNORECASE),
)

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def strip_template_control_tokens(text: str, strip_tool_tags: bool = True) -> str:
    """Remove chat-template boundary tokens and optional tool_call tags from untrusted text."""
    cleaned = str(text or "").replace("\x00", "")
    cleaned = _CONTROL_TOKEN_RE.sub("", cleaned)
    if strip_tool_tags:
        cleaned = _TOOL_TAG_RE.sub("", cleaned)
    return cleaned


def detect_role_override_attempt(text: str) -> bool:
    """Return True if text matches common prompt-injection / role-override signatures."""
    raw = str(text or "")
    return any(pat.search(raw) for pat in _ROLE_OVERRIDE_PATTERNS)


def validate_and_sanitize_messages(
    messages: Any,
    *,
    allow_mid_conversation_system: bool = False,
    max_messages: int = MAX_MESSAGES,
    max_message_chars: int = MAX_MESSAGE_CHARS,
    max_total_chars: int = MAX_TOTAL_CHARS,
) -> List[Dict[str, Any]]:
    """Validate chat messages list, enforce role boundaries, bounds, and strip control tokens."""
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("messages must be a non-empty list")
    if len(messages) > max_messages:
        raise ValueError(f"messages count exceeds maximum of {max_messages}")

    sanitized: List[Dict[str, Any]] = []
    total_chars = 0

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"message at index {idx} must be an object")

        role = msg.get("role")
        if not isinstance(role, str) or role not in ALLOWED_ROLES:
            raise ValueError(f"unsupported role at index {idx}")

        if role == "system" and idx > 0 and not allow_mid_conversation_system:
            raise ValueError("system role is only permitted as the first message")

        content = msg.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ValueError(f"message content at index {idx} must be a string")

        if len(content) > max_message_chars:
            raise ValueError(f"message at index {idx} exceeds maximum length of {max_message_chars} chars")

        # Strip template boundary tokens; also strip <tool_call> tags from user/system/tool roles
        cleaned_content = strip_template_control_tokens(
            content,
            strip_tool_tags=(role != "assistant"),
        )
        total_chars += len(cleaned_content)
        if total_chars > max_total_chars:
            raise ValueError(f"total conversation size exceeds maximum of {max_total_chars} chars")

        entry: Dict[str, Any] = {"role": role, "content": cleaned_content}
        if role == "tool" and isinstance(msg.get("name"), str) and _TOOL_NAME_RE.match(msg["name"]):
            entry["name"] = msg["name"]
        if role == "assistant" and isinstance(msg.get("tool_calls"), list):
            entry["tool_calls"] = msg["tool_calls"][:MAX_TOOLS]

        sanitized.append(entry)

    return sanitized


def validate_generation_params(
    *,
    temperature: Any,
    top_p: Any,
    max_tokens: Any,
) -> Tuple[float, float, int]:
    """Validate and bound generation hyperparameters against DoS / NaN crashes."""
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a finite number")
    temp_f = float(temperature)
    if not math.isfinite(temp_f) or temp_f < TEMPERATURE_MIN or temp_f > TEMPERATURE_MAX:
        raise ValueError(f"temperature must be between {TEMPERATURE_MIN} and {TEMPERATURE_MAX}")

    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise ValueError("top_p must be a finite number")
    top_p_f = float(top_p)
    if not math.isfinite(top_p_f) or top_p_f < TOP_P_MIN or top_p_f > TOP_P_MAX:
        raise ValueError(f"top_p must be between {TOP_P_MIN} and {TOP_P_MAX}")

    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError("max_tokens must be an integer")
    if max_tokens < MAX_TOKENS_MIN or max_tokens > MAX_TOKENS_MAX:
        raise ValueError(f"max_tokens must be between {MAX_TOKENS_MIN} and {MAX_TOKENS_MAX}")

    return temp_f, top_p_f, max_tokens


def validate_tools(tools: Any, max_tools: int = MAX_TOOLS) -> List[Dict[str, Any]]:
    """Validate OpenAI-compatible function tool declarations."""
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    if len(tools) > max_tools:
        raise ValueError(f"tools count exceeds maximum of {max_tools}")

    validated: List[Dict[str, Any]] = []
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tool at index {idx} must be an object")
        if tool.get("type") != "function":
            raise ValueError(f"tool at index {idx} must have type 'function'")
        fn = tool.get("function")
        if not isinstance(fn, dict):
            raise ValueError(f"tool at index {idx} missing function object")
        name = fn.get("name")
        if not isinstance(name, str) or not _TOOL_NAME_RE.match(name):
            raise ValueError(f"tool at index {idx} has invalid function name")
        validated.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(fn.get("description", ""))[:512],
                    "parameters": fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {},
                },
            }
        )
    return validated


def extract_open_thinking(open_thinking: bool, chat_template_kwargs: Optional[Dict[str, Any]]) -> bool:
    """Safely extract thinking flag from explicit field or allowlisted chat_template_kwargs keys."""
    if bool(open_thinking):
        return True
    if isinstance(chat_template_kwargs, dict):
        for key in ALLOWED_TEMPLATE_KWARGS:
            if bool(chat_template_kwargs.get(key, False)):
                return True
    return False


def verify_bearer_token(authorization_header: Optional[str], expected_api_key: Optional[str]) -> bool:
    """Constant-time Bearer token verification. Open when expected_api_key is unset."""
    expected = (expected_api_key or "").strip()
    if not expected:
        return True
    header = (authorization_header or "").strip()
    if not header.lower().startswith("bearer "):
        return False
    provided = header[7:].strip()
    if not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _is_private_or_reserved_ip(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_link_local
        or ip.is_loopback
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_api_base_url(url: str, *, allow_local: bool = True) -> str:
    """Validate an API base URL against SSRF, embedded credentials, and insecure remote HTTP."""
    raw = str(url or "").strip()
    if not raw:
        raise ValueError("API base URL must not be empty")
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("API base URL must use http or https scheme")
    if not parsed.hostname:
        raise ValueError("API base URL must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("API base URL must not contain embedded credentials")

    host = parsed.hostname.lower()
    is_loopback = host in _LOOPBACK_HOSTS

    if is_loopback:
        if not allow_local:
            raise ValueError("Loopback API base URLs are disabled")
        return raw.rstrip("/")

    if parsed.scheme != "https":
        raise ValueError("Remote API base URLs must use HTTPS")

    if host.endswith((".local", ".internal")) or _is_private_or_reserved_ip(host):
        raise ValueError("Private, link-local, or reserved remote hosts are blocked")

    return raw.rstrip("/")


def redact_sensitive_text(text: str) -> str:
    """Redact API keys, Bearer tokens, and URL credentials from logs/errors."""
    s = str(text or "")
    s = re.sub(r"(https?://)([^/\s:@]+:[^/\s@]+@)", r"\1***@", s, flags=re.IGNORECASE)
    s = re.sub(
        r"([?&](?:api_key|apikey|token|access_token|secret|password|key)=)[^&\s]+",
        r"\1***",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\b(Bearer\s+)[A-Za-z0-9._~+/=-]{6,}", r"\1[REDACTED]", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(?:sk|xai|gsk|AIza)[-_A-Za-z0-9]{8,}\b", "[REDACTED_KEY]", s)
    return s


def safe_error_message(exc: Exception, default_message: str = "Internal generation error") -> str:
    """Return a generic safe error message without leaking stack traces, paths, or secrets."""
    _ = redact_sensitive_text(str(exc))
    return default_message


def escape_html_text(text: str) -> str:
    """Escape arbitrary user/model text before embedding in Streamlit unsafe_allow_html blocks."""
    return html.escape(str(text or ""), quote=True)
