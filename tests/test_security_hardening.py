import ast
import asyncio
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import serve_openai_api
from scripts.api_security import (
    is_loopback_host,
    validate_server_bind_security,
    verify_bearer_token,
)


def _load_web_demo_with_stubs(enable_thinking: bool = False):
    """Import scripts/web_demo.py with lightweight stdlib stubs (no streamlit/torch needed)."""
    class _SessionState(dict):
        def __getattr__(self, name):
            if name in self:
                return self[name]
            raise AttributeError(name)

        def __setattr__(self, name, value):
            self[name] = value

    class _NoOpContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _SidebarStub:
        def selectbox(self, label, options, index=0):
            return options[index]

        def markdown(self, *args, **kwargs):
            return None

        def radio(self, label, options, index=0, **kwargs):
            return options[index]

        def slider(self, label, min_val, max_val, value, **kwargs):
            return value

        def checkbox(self, label, value=False, **kwargs):
            return value

        def expander(self, *args, **kwargs):
            return _NoOpContext()

    st_mod = types.ModuleType("streamlit")
    st_mod.session_state = _SessionState(lang="en", enable_thinking=enable_thinking)
    st_mod.sidebar = _SidebarStub()
    st_mod.set_page_config = lambda *args, **kwargs: None
    st_mod.markdown = lambda *args, **kwargs: None
    st_mod.caption = lambda *args, **kwargs: None
    st_mod.checkbox = lambda *args, **kwargs: False
    st_mod.cache_resource = lambda fn: fn

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed=lambda s: None,
        manual_seed_all=lambda s: None,
    )
    torch_mod.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(deterministic=True, benchmark=False)
    )
    torch_mod.manual_seed = lambda s: None
    torch_mod.float16 = "float16"
    torch_mod.no_grad = lambda: (lambda fn: fn)

    torch_dist_mod = types.ModuleType("torch.distributed")
    torch_dist_mod.is_initialized = lambda: False

    torch_nn_mod = types.ModuleType("torch.nn")
    torch_nn_parallel_mod = types.ModuleType("torch.nn.parallel")
    torch_nn_parallel_mod.DistributedDataParallel = type("DistributedDataParallel", (), {})

    torch_utils_mod = types.ModuleType("torch.utils")
    torch_utils_data_mod = types.ModuleType("torch.utils.data")
    torch_utils_data_mod.Sampler = object

    np_mod = types.ModuleType("numpy")
    np_mod.random = types.SimpleNamespace(seed=lambda s: None)

    recorded_calls: List[Dict[str, Any]] = []

    class _DummyLoadedModel:
        def half(self):
            return self

        def eval(self):
            return self

        def to(self, device):
            return self

    class _AutoModelStub:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            recorded_calls.append({"target": "AutoModelForCausalLM", "path": model_path, "kwargs": kwargs})
            return _DummyLoadedModel()

    class _AutoTokenizerStub:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            recorded_calls.append({"target": "AutoTokenizer", "path": model_path, "kwargs": kwargs})
            return object()

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoModelForCausalLM = _AutoModelStub
    transformers_mod.AutoTokenizer = _AutoTokenizerStub
    transformers_mod.AutoModel = _AutoModelStub
    transformers_mod.AutoModelForSequenceClassification = _AutoModelStub
    transformers_mod.TextIteratorStreamer = object

    trainer_utils_stub = types.ModuleType("trainer.trainer_utils")
    trainer_utils_stub.safe_math_eval = lambda expr: 0

    stubs = {
        "streamlit": st_mod,
        "torch": torch_mod,
        "torch.distributed": torch_dist_mod,
        "torch.nn": torch_nn_mod,
        "torch.nn.parallel": torch_nn_parallel_mod,
        "torch.utils": torch_utils_mod,
        "torch.utils.data": torch_utils_data_mod,
        "numpy": np_mod,
        "transformers": transformers_mod,
        "trainer.trainer_utils": trainer_utils_stub,
    }

    spec = importlib.util.spec_from_file_location("_test_web_demo_isolated", REPO_ROOT / "scripts" / "web_demo.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module, st_mod, recorded_calls


def _load_trainer_utils_with_stubs(torch_load_recorder):
    """Import trainer/trainer_utils.py with lightweight stubs to unit-test checkpoint loading."""
    torch_mod = types.ModuleType("torch")
    torch_mod.float16 = "float16"
    torch_mod.no_grad = lambda: (lambda fn: fn)
    torch_mod.load = torch_load_recorder

    torch_dist_mod = types.ModuleType("torch.distributed")
    torch_dist_mod.is_initialized = lambda: False

    torch_nn_mod = types.ModuleType("torch.nn")
    torch_nn_parallel_mod = types.ModuleType("torch.nn.parallel")
    torch_nn_parallel_mod.DistributedDataParallel = type("DistributedDataParallel", (), {})

    torch_utils_mod = types.ModuleType("torch.utils")
    torch_utils_data_mod = types.ModuleType("torch.utils.data")
    torch_utils_data_mod.Sampler = object

    np_mod = types.ModuleType("numpy")
    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **kw: (a, kw))
    transformers_mod.AutoModel = types.SimpleNamespace(from_pretrained=lambda *a, **kw: types.SimpleNamespace(to=lambda d: types.SimpleNamespace(eval=lambda: (a, kw))))
    transformers_mod.AutoModelForSequenceClassification = object

    stubs = {
        "torch": torch_mod,
        "torch.distributed": torch_dist_mod,
        "torch.nn": torch_nn_mod,
        "torch.nn.parallel": torch_nn_parallel_mod,
        "torch.utils": torch_utils_mod,
        "torch.utils.data": torch_utils_data_mod,
        "numpy": np_mod,
        "transformers": transformers_mod,
    }

    spec = importlib.util.spec_from_file_location("_test_trainer_utils_isolated", REPO_ROOT / "trainer" / "trainer_utils.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def _post_asgi(app, path: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None):
    body_bytes = json.dumps(payload).encode("utf-8")
    raw_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body_bytes)).encode("ascii")),
    ]
    for k, v in (headers or {}).items():
        raw_headers.append((k.lower().encode("latin-1"), v.encode("latin-1")))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": raw_headers,
        "client": ("127.0.0.1", 54321),
        "server": ("testserver", 80),
    }
    sent = False
    status_code = 500
    chunks: List[bytes] = []

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    asyncio.run(app(scope, receive, send))
    return status_code, b"".join(chunks).decode("utf-8", errors="replace")


class TestWebDemoXssAndRemoteCode(unittest.TestCase):
    def test_process_assistant_content_escapes_xss_in_plain_think_and_tool_blocks(self):
        web_demo, st_mod, _ = _load_web_demo_with_stubs(enable_thinking=False)

        # 1. Plain assistant response with XSS payloads
        plain_payload = 'Hello\n<script>alert(1)</script><img src=x onerror=alert(1)>'
        rendered_plain = web_demo.process_assistant_content(plain_payload)
        self.assertNotIn("<script>", rendered_plain)
        self.assertNotIn("<img ", rendered_plain)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered_plain)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", rendered_plain)
        self.assertIn("Hello<br>", rendered_plain)

        # 2. <think>...</think> with XSS inside think AND outside in answer
        think_payload = (
            '<think>plan <script>alert(1)</script> & <img src=x onerror=alert(1)></think>\n'
            'Final answer: <script>alert(2)</script><img src=x onerror=alert(2)>'
        )
        rendered_think = web_demo.process_assistant_content(think_payload)
        self.assertIn("<details open", rendered_think)
        self.assertNotIn("<script>", rendered_think)
        self.assertNotIn("<img ", rendered_think)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered_think)
        self.assertIn("&lt;script&gt;alert(2)&lt;/script&gt;", rendered_think)
        self.assertIn("&lt;img src=x onerror=alert(2)&gt;", rendered_think)

        # 3. Leading </think> without opening <think>
        no_open_think = 'thought <img src=x onerror=alert(1)></think><script>alert(3)</script>'
        rendered_no_open = web_demo.process_assistant_content(no_open_think)
        self.assertIn("<details open", rendered_no_open)
        self.assertNotIn("<script>", rendered_no_open)
        self.assertNotIn("<img ", rendered_no_open)
        self.assertIn("&lt;script&gt;alert(3)&lt;/script&gt;", rendered_no_open)

        # 4. In-progress streaming <think> (unclosed)
        unclosed_think = 'prefix <img src=x onerror=1><think>working <script>alert(4)</script>'
        rendered_unclosed = web_demo.process_assistant_content(unclosed_think, is_streaming=True)
        self.assertIn("思考中...", rendered_unclosed)
        self.assertNotIn("<script>", rendered_unclosed)
        self.assertNotIn("<img ", rendered_unclosed)

        # 5. Streaming heuristic when enable_thinking=True
        st_mod.session_state["enable_thinking"] = True
        heuristic_stream = 'thinking step <img src=x onerror=1>\n\n你好 <script>alert(5)</script>'
        rendered_heuristic = web_demo.process_assistant_content(heuristic_stream, is_streaming=True)
        self.assertIn("<details open", rendered_heuristic)
        self.assertNotIn("<script>", rendered_heuristic)
        self.assertNotIn("<img ", rendered_heuristic)
        self.assertIn("&lt;script&gt;alert(5)&lt;/script&gt;", rendered_heuristic)

        # 6. <tool_call> and <tool_result> with XSS in name, args, and surrounding text
        st_mod.session_state["enable_thinking"] = False
        tool_payload = (
            'Before <img src=x onerror=alert(1)>\n'
            '<tool_call>{"name": "<script>alert(1)</script>", "arguments": {"q": "<img src=x onerror=alert(1)>"}}</tool_call>\n'
            '<tool_result>{"name": "<script>x</script>", "result": {"out": "<img src=x onerror=alert(1)>"}}</tool_result>\n'
            'After <script>alert(6)</script>'
        )
        rendered_tool = web_demo.process_assistant_content(tool_payload)
        self.assertIn("ToolCalling", rendered_tool)
        self.assertIn("ToolCalled", rendered_tool)
        self.assertNotIn("<script>", rendered_tool)
        self.assertNotIn("<img ", rendered_tool)
        self.assertIn("&lt;script&gt;alert(6)&lt;/script&gt;", rendered_tool)

        # 7. Malformed & unclosed <tool_call> tags
        malformed_tool = '<tool_call>invalid <script>alert(7)</script></tool_call><tool_call><img src=x onerror=alert(8)>'
        rendered_malformed = web_demo.process_assistant_content(malformed_tool)
        self.assertNotIn("<script>", rendered_malformed)
        self.assertNotIn("<img ", rendered_malformed)
        self.assertIn("&lt;script&gt;alert(7)&lt;/script&gt;", rendered_malformed)
        self.assertIn("&lt;img src=x onerror=alert(8)&gt;", rendered_malformed)

    def test_web_demo_load_model_tokenizer_trust_remote_code_opt_in(self):
        web_demo, _, calls = _load_web_demo_with_stubs()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MINIMIND_TRUST_REMOTE_CODE", None)
            web_demo.load_model_tokenizer("dummy-model")
        self.assertEqual(len(calls), 2)
        self.assertFalse(calls[0]["kwargs"]["trust_remote_code"])
        self.assertFalse(calls[1]["kwargs"]["trust_remote_code"])

        calls.clear()
        with mock.patch.dict(os.environ, {"MINIMIND_TRUST_REMOTE_CODE": "1"}, clear=False):
            web_demo.load_model_tokenizer("dummy-model")
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0]["kwargs"]["trust_remote_code"])
        self.assertTrue(calls[1]["kwargs"]["trust_remote_code"])


class TestStaticSecurityAudit(unittest.TestCase):
    def _iter_target_python_files(self) -> List[Path]:
        files = [REPO_ROOT / "eval_llm.py"]
        for subdir in ("model", "scripts", "trainer"):
            files.extend(sorted((REPO_ROOT / subdir).glob("*.py")))
        return files

    def test_no_bare_torch_load_without_weights_only(self):
        violations: List[str] = []
        for path in self._iter_target_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                ):
                    kw = {k.arg: k.value for k in node.keywords if k.arg is not None}
                    if "weights_only" not in kw:
                        violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} missing weights_only=")
                    elif isinstance(kw["weights_only"], ast.Constant) and kw["weights_only"].value is False:
                        violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} hardcoded weights_only=False")
        self.assertEqual(violations, [], f"Unsafe torch.load calls found: {violations}")

    def test_no_hardcoded_trust_remote_code_true(self):
        violations: List[str] = []
        for path in self._iter_target_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if (
                            kw.arg == "trust_remote_code"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                        ):
                            violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} hardcoded trust_remote_code=True")
        self.assertEqual(violations, [], f"Hardcoded trust_remote_code=True found: {violations}")

    def test_trainer_resume_checkpoint_weights_only_default_and_opt_in(self):
        recorded: List[Dict[str, Any]] = []

        def _fake_torch_load(path, map_location="cpu", weights_only=None):
            recorded.append({"path": path, "map_location": map_location, "weights_only": weights_only})
            return {"step": 1}

        trainer_utils = _load_trainer_utils_with_stubs(_fake_torch_load)

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MINIMIND_ALLOW_UNSAFE_CHECKPOINT", None)
            trainer_utils.load_resume_checkpoint("ckpt_resume.pth")
        self.assertEqual(recorded[-1]["weights_only"], True)

        with mock.patch.dict(os.environ, {"MINIMIND_ALLOW_UNSAFE_CHECKPOINT": "1"}, clear=False):
            trainer_utils.load_resume_checkpoint("ckpt_resume.pth")
        self.assertEqual(recorded[-1]["weights_only"], False)


class TestNonLoopbackFailClosedAuth(unittest.TestCase):
    def test_loopback_detection_and_bind_validation(self):
        for host in ("127.0.0.1", "localhost", "::1", "[::1]", " LOCALHOST "):
            self.assertTrue(is_loopback_host(host), f"Expected loopback for {host!r}")
        for host in ("0.0.0.0", "::", "192.168.1.10", "10.0.0.5", "api.example.com", "", None):
            self.assertFalse(is_loopback_host(host), f"Expected non-loopback for {host!r}")

        # Non-loopback without API key and without explicit opt-in -> refused
        with self.assertRaises(ValueError):
            validate_server_bind_security("0.0.0.0", None)
        with self.assertRaises(ValueError):
            validate_server_bind_security("::", "")
        with self.assertRaises(ValueError):
            validate_server_bind_security("192.168.1.25", "   ")

        # Non-loopback WITH API key -> allowed and requires auth
        self.assertTrue(validate_server_bind_security("0.0.0.0", "strong-key"))

        # Non-loopback WITH explicit opt-in -> allowed without auth
        self.assertFalse(validate_server_bind_security("0.0.0.0", None, allow_unauthenticated=True))

        # Loopback without API key -> allowed, auth optional
        self.assertFalse(validate_server_bind_security("127.0.0.1", None))
        # Loopback with API key -> allowed, auth required
        self.assertTrue(validate_server_bind_security("localhost", "local-key"))

    def test_verify_bearer_token_fail_closed_when_require_auth_true(self):
        # Backwards-compatible default (require_auth=False)
        self.assertTrue(verify_bearer_token(None, None))
        # Fail-closed when require_auth=True and expected key is unset/blank
        self.assertFalse(verify_bearer_token(None, None, require_auth=True))
        self.assertFalse(verify_bearer_token("Bearer any-token", "", require_auth=True))
        self.assertFalse(verify_bearer_token("Bearer wrong", "expected", require_auth=True))
        self.assertTrue(verify_bearer_token("Bearer expected", "expected", require_auth=True))

    def test_serve_openai_api_configure_auth_and_fail_closed_endpoint(self):
        orig_api_key = serve_openai_api.API_KEY
        orig_require_auth = serve_openai_api.REQUIRE_AUTH
        try:
            serve_openai_api.API_KEY = None
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MINIMIND_API_KEY", None)
                with self.assertRaises(ValueError):
                    serve_openai_api.configure_server_auth("0.0.0.0", None)
                self.assertFalse(
                    serve_openai_api.configure_server_auth("0.0.0.0", None, allow_unauthenticated=True)
                )
                self.assertTrue(serve_openai_api.configure_server_auth("0.0.0.0", "cli-secret"))

                # Endpoint fails closed with 401 if REQUIRE_AUTH=True even when API_KEY is unset
                serve_openai_api.REQUIRE_AUTH = True
                status, _ = _post_asgi(
                    serve_openai_api.app,
                    "/v1/chat/completions",
                    {"model": "minimind", "messages": [{"role": "user", "content": "hi"}], "stream": False},
                )
                self.assertEqual(status, 401)
        finally:
            serve_openai_api.API_KEY = orig_api_key
            serve_openai_api.REQUIRE_AUTH = orig_require_auth


class TestRequirementsHygiene(unittest.TestCase):
    def test_no_duplicate_entries_in_requirements(self):
        req_lines = [
            line.strip()
            for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        pkg_names = [line.split("==", 1)[0].strip().lower() for line in req_lines]
        self.assertEqual(pkg_names.count("jsonlines"), 1)
        self.assertEqual(len(pkg_names), len(set(pkg_names)), f"Duplicate dependencies in requirements.txt: {pkg_names}")


if __name__ == "__main__":
    unittest.main()
