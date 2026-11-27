import argparse
import json
import logging
import os
import re
import sys
import time
import warnings
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Optional

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scripts.api_security import (
    MAX_MESSAGES,
    MAX_MODEL_NAME_LEN,
    MAX_TOKENS_MAX,
    MAX_TOKENS_MIN,
    MAX_TOOLS,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    TOP_P_MAX,
    TOP_P_MIN,
    extract_open_thinking,
    redact_sensitive_text,
    safe_error_message,
    validate_and_sanitize_messages,
    validate_generation_params,
    validate_server_bind_security,
    validate_tools,
    verify_bearer_token,
)

try:
    import torch
except ImportError:
    torch = None

try:
    import uvicorn
except ImportError:
    uvicorn = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

    class TextStreamer:  # type: ignore[no-redef]
        def __init__(self, tokenizer=None, skip_prompt=True, skip_special_tokens=True):
            self.tokenizer = tokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

try:
    from model.model_lora import apply_lora, load_lora
except ImportError:
    apply_lora = None
    load_lora = None

warnings.filterwarnings('ignore')
logger = logging.getLogger("minimind.api")

app = FastAPI()

# Runtime globals populated by init_model() or test fixtures
device = "cpu"
model = None
tokenizer = None
API_KEY: Optional[str] = os.environ.get("MINIMIND_API_KEY", "").strip() or None
REQUIRE_AUTH: bool = False


def get_configured_api_key() -> Optional[str]:
    if API_KEY:
        return API_KEY
    env_key = os.environ.get("MINIMIND_API_KEY", "").strip()
    return env_key or None


def configure_server_auth(
    host: str,
    api_key: Optional[str] = None,
    *,
    allow_unauthenticated: bool = False,
) -> bool:
    """Validate server bind host/auth settings and return whether auth is required."""
    effective_key = (api_key or "").strip() or get_configured_api_key()
    return validate_server_bind_security(
        host,
        effective_key,
        allow_unauthenticated=allow_unauthenticated,
    )


def init_model(args):
    if torch is None or AutoTokenizer is None:
        raise RuntimeError("torch and transformers are required to load MiniMind weights")
    tok = AutoTokenizer.from_pretrained(args.load_from)
    if os.path.basename(os.path.normpath(args.load_from)) == 'model':
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        mdl = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        mdl.load_state_dict(torch.load(ckp, map_location=device, weights_only=True), strict=True)
        if args.lora_weight != 'None':
            apply_lora(mdl, rank=args.lora_rank)
            load_lora(mdl, f'../{args.save_dir}/{args.lora_weight}_{args.hidden_size}{moe_suffix}.pth')
    else:
        trust_remote_code = bool(getattr(args, "trust_remote_code", False)) or (os.getenv("MINIMIND_TRUST_REMOTE_CODE") == "1")
        mdl = AutoModelForCausalLM.from_pretrained(
            args.load_from,
            trust_remote_code=trust_remote_code,
        )
    print(f'MiniMind模型参数量: {sum(p.numel() for p in mdl.parameters()) / 1e6:.2f} M(illion)')
    return mdl.half().eval().to(device), tok


class ChatRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=MAX_MODEL_NAME_LEN)
    messages: List[Dict[str, Any]] = Field(..., min_length=1, max_length=MAX_MESSAGES)
    temperature: float = Field(default=0.7, ge=TEMPERATURE_MIN, le=TEMPERATURE_MAX, allow_inf_nan=False)
    top_p: float = Field(default=0.92, ge=TOP_P_MIN, le=TOP_P_MAX, allow_inf_nan=False)
    max_tokens: int = Field(default=2048, ge=MAX_TOKENS_MIN, le=MAX_TOKENS_MAX)
    stream: bool = True
    tools: List[Dict[str, Any]] = Field(default_factory=list, max_length=MAX_TOOLS)
    open_thinking: bool = False
    chat_template_kwargs: Optional[Dict[str, Any]] = None

    def get_open_thinking(self) -> bool:
        """兼容多种方式开启 thinking (restricted to allowlisted keys)"""
        return extract_open_thinking(self.open_thinking, self.chat_template_kwargs)


class CustomStreamer(TextStreamer):
    def __init__(self, tok, queue):
        super().__init__(tok, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tok

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def parse_response(text):
    reasoning_content = None
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    elif '</think>' in text:
        parts = text.split('</think>', 1)
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ''
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            tool_calls.append({"id": f"call_{int(time.time())}_{i}", "type": "function", "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}})
        except Exception:
            pass
    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    return text.strip(), reasoning_content, tool_calls or None


def generate_stream_response(messages, temperature, top_p, max_tokens, tools=None, open_thinking=False):
    try:
        if model is None or tokenizer is None:
            raise RuntimeError("Model is not initialized")

        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools or None, open_thinking=open_thinking)
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True)
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)

        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            try:
                model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer
                )
            except Exception as e:
                logger.warning("MiniMind stream generation error: %s", redact_sensitive_text(str(e)))
                queue.put({"error": safe_error_message(e, "Stream generation failed")})
                queue.put(None)

        Thread(target=_generate, daemon=True).start()

        pending = ""
        in_tool_call = False
        tool_count = 0
        thinking_ended = not bool(open_thinking)
        strip_newlines = False

        while True:
            text = queue.get()
            if text is None:
                break
            if isinstance(text, dict):
                yield json.dumps(text, ensure_ascii=False)
                continue
            pending += text

            while pending:
                if strip_newlines:
                    pending = pending.lstrip('\n')
                    if not pending:
                        break
                    strip_newlines = False
                if in_tool_call:
                    end = pending.find('</tool_call>')
                    if end < 0:
                        break
                    end += len('</tool_call>')
                    block, pending = pending[:end], pending[end:]
                    _, _, tool_calls = parse_response(block)
                    if tool_calls:
                        call = tool_calls[0]
                        call['index'] = tool_count
                        call['id'] = f"call_{int(time.time())}_{tool_count}"
                        tool_count += 1
                        yield json.dumps({"choices": [{"delta": {"tool_calls": [call]}}]}, ensure_ascii=False)
                    else:
                        yield json.dumps({"choices": [{"delta": {"content": block}}]}, ensure_ascii=False)
                    in_tool_call = False
                    continue

                marker = '<tool_call>' if thinking_ended else '</think>'
                field = 'content' if thinking_ended else 'reasoning_content'
                pos = pending.find(marker)
                if pos < 0:
                    keep = next((n for n in range(len(marker) - 1, 0, -1)
                                 if pending.endswith(marker[:n])), 0)
                    end = len(pending) - keep
                    if end:
                        yield json.dumps({"choices": [{"delta": {field: pending[:end]}}]}, ensure_ascii=False)
                        pending = pending[end:]
                    break
                if pos:
                    yield json.dumps({"choices": [{"delta": {field: pending[:pos]}}]}, ensure_ascii=False)
                pending = pending[pos:]
                if thinking_ended:
                    in_tool_call = True
                else:
                    pending = pending[len(marker):]
                    thinking_ended = True
                    strip_newlines = True

        if pending:
            field = 'content' if thinking_ended else 'reasoning_content'
            yield json.dumps({"choices": [{"delta": {field: pending}}]}, ensure_ascii=False)
        yield json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_count else "stop"}]}, ensure_ascii=False)

    except Exception as e:
        logger.warning("MiniMind stream error: %s", redact_sensitive_text(str(e)))
        yield json.dumps({"error": safe_error_message(e, "Stream generation failed")})


@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatRequest,
    authorization: Optional[str] = Header(default=None),
):
    expected_key = get_configured_api_key()
    if not verify_bearer_token(authorization, expected_key, require_auth=REQUIRE_AUTH):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        sanitized_messages = validate_and_sanitize_messages(request.messages)
        temperature, top_p, max_tokens = validate_generation_params(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        sanitized_tools = validate_tools(request.tools)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model service is not initialized")

    try:
        if request.stream:
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=sanitized_messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    tools=sanitized_tools,
                    open_thinking=request.get_open_thinking()
                )),
                media_type="text/event-stream",
                headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"},
            )
        else:
            new_prompt = tokenizer.apply_chat_template(
                sanitized_messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=sanitized_tools or None,
                open_thinking=request.get_open_thinking()
            )
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True)
            if hasattr(inputs, "to"):
                inputs = inputs.to(device)

            def _run_generate():
                return model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=top_p,
                    temperature=temperature
                )

            if torch is not None:
                with torch.no_grad():
                    generated_ids = _run_generate()
            else:
                generated_ids = _run_generate()

            prompt_len = inputs["input_ids"].shape[1] if hasattr(inputs["input_ids"], "shape") else len(inputs["input_ids"][0])
            answer = tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True)
            content, reasoning_content, tool_calls = parse_response(answer)
            message = {"role": "assistant", "content": content}
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop"
                    }
                ]
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("MiniMind completion error: %s", redact_sensitive_text(str(e)))
        raise HTTPException(status_code=500, detail=safe_error_message(e, "Internal generation error"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for MiniMind")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, dpo, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--lora_rank', default=16, type=int, help="LoRA rank，需与训练时一致（默认16）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--device', default='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu', type=str, help="运行设备")
    parser.add_argument('--host', default='127.0.0.1', type=str, help="服务监听地址（默认 127.0.0.1，如需外部访问请显式传 0.0.0.0 并配置 MINIMIND_API_KEY）")
    parser.add_argument('--port', default=8998, type=int, help="服务监听端口")
    parser.add_argument('--api_key', default='', type=str, help="可选 Bearer API Key（也可通过 MINIMIND_API_KEY 环境变量设置）")
    parser.add_argument('--trust_remote_code', default=False, action='store_true', help="仅在信任 HuggingFace 模型仓库时启用远程代码执行")
    parser.add_argument('--allow-unauthenticated', '--allow_unauthenticated', dest='allow_unauthenticated', default=False, action='store_true', help="显式允许在非回环地址上不配置 API Key 启动（不推荐）")
    args = parser.parse_args()
    device = args.device
    if args.api_key.strip():
        API_KEY = args.api_key.strip()
    REQUIRE_AUTH = configure_server_auth(
        args.host,
        API_KEY,
        allow_unauthenticated=args.allow_unauthenticated,
    )
    model, tokenizer = init_model(args)
    if uvicorn is None:
        raise RuntimeError("uvicorn is required to start the server")
    uvicorn.run(app, host=args.host, port=args.port)
