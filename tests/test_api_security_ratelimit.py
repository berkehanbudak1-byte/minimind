"""Stdlib unittest suite for SlidingWindowAuthRateLimiter, DPAPI secret helpers, and serve_openai_api wiring."""
import asyncio
import base64
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import api_security, serve_openai_api
from scripts.api_security import (
    DPAPI_PREFIX,
    DPAPIError,
    SlidingWindowAuthRateLimiter,
    is_dpapi_secret,
    redact_sensitive_text,
    resolve_dpapi_secret,
    seal_dpapi_secret,
    unseal_dpapi_secret,
    verify_bearer_token,
)


def _make_fake_dpapi_runner():
    calls = []

    def _xor(data: bytes, mask: bytes) -> bytes:
        return bytes(b ^ mask[i % len(mask)] for i, b in enumerate(data))

    def runner(cmd, *, capture_output=True, text=True, env=None, timeout=None, check=False, shell=False):
        calls.append({"cmd": list(cmd), "env": dict(env or {}), "shell": shell, "timeout": timeout})
        script = cmd[-1]
        in_bytes = base64.b64decode(env["_DPAPI_INPUT_B64"], validate=True)
        scope = env.get("_DPAPI_SCOPE", "CurrentUser").encode("utf-8")
        entropy = base64.b64decode(env["_DPAPI_ENTROPY_B64"], validate=True) if env.get("_DPAPI_ENTROPY_B64") else b"default"
        mask = b"MINIMIND-DPAPI:" + scope + b":" + entropy
        if "::Protect(" in script:
            out_bytes = b"M1|" + _xor(in_bytes, mask)
        elif "::Unprotect(" in script:
            if not in_bytes.startswith(b"M1|"):
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="invalid blob")
            out_bytes = _xor(in_bytes[3:], mask)
        else:
            return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="unknown script")
        return subprocess.CompletedProcess(cmd, 0, stdout=base64.b64encode(out_bytes).decode("ascii"), stderr="")

    return runner, calls


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, tools=None, open_thinking=False):
        return "prompt"

    def __call__(self, text, return_tensors="pt", truncation=True):
        return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

    def decode(self, tokens, skip_special_tokens=True):
        return "Safe MiniMind reply"


class _DummyModel:
    def generate(self, input_ids, **kwargs):
        return [[1, 2, 3, 4]]


async def _asgi_post_json(
    asgi_app,
    path: str,
    payload: dict,
    *,
    headers: dict | None = None,
    client_ip: str = "198.51.100.10",
) -> tuple[int, dict[str, str], dict]:
    """Invoke a FastAPI ASGI endpoint directly via stdlib asyncio (no TestClient/httpx needed)."""
    body = json.dumps(payload).encode("utf-8")
    raw_headers = [
        (b"host", b"testserver"),
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    for k, v in (headers or {}).items():
        raw_headers.append((k.lower().encode("latin-1"), v.encode("latin-1")))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": raw_headers,
        "client": (client_ip, 51234),
        "server": ("testserver", 443),
    }

    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    status_code = 0
    resp_headers: dict[str, str] = {}
    chunks: list[bytes] = []

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
            for k, v in message.get("headers", []):
                resp_headers[k.decode("latin-1").lower()] = v.decode("latin-1")
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await asgi_app(scope, receive, send)
    raw_body = b"".join(chunks).decode("utf-8")
    parsed = json.loads(raw_body) if raw_body else {}
    return status_code, resp_headers, parsed


class TestSlidingWindowAuthRateLimiter(unittest.TestCase):
    def test_sliding_window_lockout_recovery_and_ip_isolation(self):
        now = [500.0]
        limiter = SlidingWindowAuthRateLimiter(
            max_failures=3,
            window_seconds=30.0,
            max_tracked_ips=10,
            clock=lambda: now[0],
        )

        # Invalid constructor bounds rejected
        with self.assertRaises(ValueError):
            SlidingWindowAuthRateLimiter(max_failures=0)
        with self.assertRaises(ValueError):
            SlidingWindowAuthRateLimiter(window_seconds=0)
        with self.assertRaises(ValueError):
            SlidingWindowAuthRateLimiter(max_tracked_ips=0)

        self.assertEqual(limiter.check_allowed("198.51.100.1"), (True, 0))
        self.assertEqual(limiter.record_failure("198.51.100.1"), 1)
        now[0] += 5.0
        self.assertEqual(limiter.record_failure("198.51.100.1"), 2)
        self.assertFalse(limiter.is_blocked("198.51.100.1"))

        now[0] += 5.0
        self.assertEqual(limiter.record_failure("198.51.100.1"), 3)
        allowed, retry_after = limiter.check_allowed("198.51.100.1")
        self.assertFalse(allowed)
        self.assertEqual(retry_after, 20)

        # Other IP remains unaffected
        self.assertEqual(limiter.check_allowed("198.51.100.2"), (True, 0))

        # Oldest failure ages out at t=530.0 -> unblocks
        now[0] += 21.0
        self.assertEqual(limiter.check_allowed("198.51.100.1"), (True, 0))

        # Success clears failure bucket
        limiter.record_success("198.51.100.1")
        self.assertEqual(len(limiter), 0)

    def test_bounded_tracked_ips_eviction(self):
        now = [100.0]
        limiter = SlidingWindowAuthRateLimiter(
            max_failures=5,
            window_seconds=60.0,
            max_tracked_ips=4,
            clock=lambda: now[0],
        )
        for i in range(25):
            limiter.record_failure(f"192.0.2.{i}")
            now[0] += 0.1
        self.assertLessEqual(len(limiter), 4)


class TestDPAPISecretHelpers(unittest.TestCase):
    def test_dpapi_seal_unseal_roundtrip_and_zero_cmd_interpolation(self):
        runner, calls = _make_fake_dpapi_runner()
        secret = "mm-key-ğüşöçıİ-$(id)\"';|secret"
        entropy = "minimind-entropy"

        sealed = seal_dpapi_secret(secret, scope="LocalMachine", entropy=entropy, runner=runner)
        self.assertTrue(is_dpapi_secret(sealed))
        self.assertTrue(sealed.startswith(DPAPI_PREFIX))
        self.assertNotIn(secret, sealed)

        unsealed = unseal_dpapi_secret(sealed, scope="LocalMachine", entropy=entropy, runner=runner)
        self.assertEqual(unsealed, secret)
        self.assertEqual(resolve_dpapi_secret(sealed, scope="LocalMachine", entropy=entropy, runner=runner), secret)
        self.assertEqual(resolve_dpapi_secret("plain-key", runner=runner), "plain-key")
        self.assertIsNone(resolve_dpapi_secret("   ", runner=runner))

        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertFalse(call["shell"])
            joined_cmd = " ".join(call["cmd"])
            self.assertNotIn(secret, joined_cmd)
            self.assertNotIn(entropy, joined_cmd)
            self.assertNotIn(sealed[len(DPAPI_PREFIX):], joined_cmd)
            self.assertIn("_DPAPI_INPUT_B64", call["env"])
            self.assertEqual(call["env"]["_DPAPI_SCOPE"], "LocalMachine")

        # Redaction scrubs dpapi:v1: payloads
        redacted = redact_sensitive_text(f"token={sealed}")
        self.assertIn("dpapi:v1:[REDACTED]", redacted)
        self.assertNotIn(sealed[len(DPAPI_PREFIX):], redacted)

    def test_dpapi_validation_non_windows_and_no_secret_leak_on_error(self):
        runner, _ = _make_fake_dpapi_runner()
        with self.assertRaises(ValueError):
            seal_dpapi_secret("", runner=runner)
        with self.assertRaises(ValueError):
            seal_dpapi_secret("abc", scope="Bogus", runner=runner)
        with self.assertRaises(ValueError):
            unseal_dpapi_secret("plain", runner=runner)
        with self.assertRaises(ValueError):
            unseal_dpapi_secret("dpapi:v1:***bad***", runner=runner)

        probe = "MINIMIND_SECRET_NEVER_LEAK"

        def fail_runner(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=f"error {probe}")

        with self.assertRaises(DPAPIError) as ctx:
            seal_dpapi_secret(probe, runner=fail_runner)
        self.assertNotIn(probe, str(ctx.exception))

        with patch.object(api_security.sys, "platform", "linux"):
            with self.assertRaises(DPAPIError):
                seal_dpapi_secret("abc")


class TestServeOpenAIAPIRateLimitAndDPAPI(unittest.TestCase):
    def setUp(self):
        self._orig_model = serve_openai_api.model
        self._orig_tok = serve_openai_api.tokenizer
        self._orig_key = serve_openai_api.API_KEY
        self._orig_limiter = serve_openai_api.auth_rate_limiter
        self._orig_env_key = os.environ.get("MINIMIND_API_KEY")
        serve_openai_api.model = _DummyModel()
        serve_openai_api.tokenizer = _DummyTokenizer()

    def tearDown(self):
        serve_openai_api.model = self._orig_model
        serve_openai_api.tokenizer = self._orig_tok
        serve_openai_api.API_KEY = self._orig_key
        serve_openai_api.auth_rate_limiter = self._orig_limiter
        if self._orig_env_key is None:
            os.environ.pop("MINIMIND_API_KEY", None)
        else:
            os.environ["MINIMIND_API_KEY"] = self._orig_env_key

    def test_open_mode_when_api_key_unset_and_strict_security_headers(self):
        serve_openai_api.API_KEY = None
        os.environ.pop("MINIMIND_API_KEY", None)
        status, headers, data = asyncio.run(
            _asgi_post_json(
                serve_openai_api.app,
                "/v1/chat/completions",
                {"model": "minimind", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["choices"][0]["message"]["content"], "Safe MiniMind reply")
        self.assertEqual(headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(headers.get("x-frame-options"), "DENY")
        csp = headers.get("content-security-policy", "")
        self.assertIn("script-src 'none'", csp)
        self.assertNotIn("'unsafe-inline'", csp)

    def test_per_ip_429_rate_limiting_and_dpapi_key_verification(self):
        now = [2000.0]
        serve_openai_api.auth_rate_limiter = SlidingWindowAuthRateLimiter(
            max_failures=2,
            window_seconds=60.0,
            max_tracked_ips=16,
            clock=lambda: now[0],
        )
        runner, _ = _make_fake_dpapi_runner()
        sealed_key = seal_dpapi_secret("mm-prod-secret-42", runner=runner)
        serve_openai_api.API_KEY = sealed_key
        payload = {"model": "minimind", "messages": [{"role": "user", "content": "hello"}], "stream": False}

        with patch.object(api_security.sys, "platform", "win32"), patch.object(
            api_security.subprocess, "run", side_effect=runner
        ):
            # 2 failed auth attempts from IP A -> 401
            for _ in range(2):
                st, _, _ = asyncio.run(
                    _asgi_post_json(
                        serve_openai_api.app,
                        "/v1/chat/completions",
                        payload,
                        headers={"Authorization": "Bearer wrong-key"},
                        client_ip="198.51.100.77",
                    )
                )
                self.assertEqual(st, 401)

            # 3rd attempt from IP A -> 429 Too Many Requests with Retry-After (even with valid token!)
            st_429, hdr_429, _ = asyncio.run(
                _asgi_post_json(
                    serve_openai_api.app,
                    "/v1/chat/completions",
                    payload,
                    headers={"Authorization": "Bearer mm-prod-secret-42"},
                    client_ip="198.51.100.77",
                )
            )
            self.assertEqual(st_429, 429)
            self.assertGreaterEqual(int(hdr_429["retry-after"]), 1)

            # Different IP B with valid token -> 200
            st_b, _, data_b = asyncio.run(
                _asgi_post_json(
                    serve_openai_api.app,
                    "/v1/chat/completions",
                    payload,
                    headers={"Authorization": "Bearer mm-prod-secret-42"},
                    client_ip="198.51.100.88",
                )
            )
            self.assertEqual(st_b, 200)
            self.assertEqual(data_b["choices"][0]["message"]["content"], "Safe MiniMind reply")

            # Advance clock past window -> IP A succeeds with valid token
            now[0] += 61.0
            st_ok, _, _ = asyncio.run(
                _asgi_post_json(
                    serve_openai_api.app,
                    "/v1/chat/completions",
                    payload,
                    headers={"Authorization": "Bearer mm-prod-secret-42"},
                    client_ip="198.51.100.77",
                )
            )
            self.assertEqual(st_ok, 200)

        # Verify constant-time helper directly too
        self.assertTrue(verify_bearer_token("Bearer mm-prod-secret-42", "mm-prod-secret-42"))
        self.assertFalse(verify_bearer_token("Bearer mm-prod-secret-43", "mm-prod-secret-42"))

    def test_corrupt_dpapi_key_fails_closed_503(self):
        serve_openai_api.auth_rate_limiter = SlidingWindowAuthRateLimiter(max_failures=5, window_seconds=60.0)
        serve_openai_api.API_KEY = "dpapi:v1:!!!bad_base64!!!"
        payload = {"model": "minimind", "messages": [{"role": "user", "content": "hello"}], "stream": False}
        st, _, data = asyncio.run(
            _asgi_post_json(
                serve_openai_api.app,
                "/v1/chat/completions",
                payload,
                headers={"Authorization": "Bearer anything"},
                client_ip="198.51.100.99",
            )
        )
        self.assertEqual(st, 503)
        self.assertNotIn("bad_base64", json.dumps(data))


if __name__ == "__main__":
    unittest.main()
