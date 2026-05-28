"""OpenAI-compatible local server backed by a ChatGPT subscription."""

from __future__ import annotations

import json
import os
import threading
import time
import warnings
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import uuid4

os.environ.setdefault("LITELLM_LOG", "ERROR")
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:*",
    category=UserWarning,
    module="pydantic.main",
)

import httpx
import litellm
from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import (
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
)

litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.turn_off_message_logging = True


FALLBACK_INSTRUCTIONS = "You are a careful assistant. Follow the user request exactly."


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            content = "\n".join(part for part in parts if part)
        normalized.append(
            {
                "role": str(message.get("role") or "user"),
                "content": str(content),
            }
        )
    return normalized


def _reasoning_effort_from_request(request: dict[str, Any]) -> str | None:
    reasoning = request.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        return str(effort) if effort else None

    effort = request.get("reasoning_effort")
    if isinstance(effort, str) and effort.strip():
        return effort.strip()
    return None


def _split_instructions_and_input(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    instructions: list[str] = []
    response_input: list[dict[str, str]] = []

    for message in _normalize_messages(messages):
        role = message["role"]
        content = message["content"]
        if role in {"system", "developer"}:
            instructions.append(content)
            continue
        response_input.append(
            {"role": role if role in {"user", "assistant"} else "user", "content": content}
        )

    return (
        "\n\n".join(part for part in instructions if part).strip() or FALLBACK_INSTRUCTIONS,
        response_input,
    )


def _chatgpt_request_body(
    model: str,
    messages: list[dict[str, Any]],
    *,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    instructions, response_input = _split_instructions_and_input(messages)
    body: dict[str, Any] = {
        "model": model.removeprefix("chatgpt/"),
        "instructions": instructions,
        "input": response_input or [{"role": "user", "content": ""}],
        "stream": True,
        "store": False,
    }
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    return body


def _chatgpt_stream_lines(
    model: str,
    messages: list[dict[str, Any]],
    *,
    reasoning_effort: str | None = None,
) -> Iterator[str]:
    authenticator = Authenticator()
    token = authenticator.get_access_token()
    api_base = authenticator.get_api_base().rstrip("/")
    headers = get_chatgpt_default_headers(
        token,
        authenticator.get_account_id(),
        ensure_chatgpt_session_id({"litellm_call_id": f"distillation-blogs-{uuid4()}"}),
    )
    with httpx.stream(
        "POST",
        f"{api_base}/responses",
        headers=headers,
        json=_chatgpt_request_body(
            model,
            messages,
            reasoning_effort=reasoning_effort,
        ),
        timeout=300,
    ) as response:
        if response.status_code >= 400:
            body = response.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"ChatGPT subscription request failed: {response.status_code} {body}")
        yield from response.iter_lines()


def _text_from_sse_lines(lines: Iterator[str]) -> str:
    parts: list[str] = []
    final_text = ""
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        if event.get("type") == "response.output_text.delta":
            parts.append(str(event.get("delta") or ""))
        elif event.get("type") == "response.output_text.done":
            final_text = str(event.get("text") or final_text)
    return "".join(parts) or final_text


def collect_chatgpt_text(
    model: str,
    messages: list[dict[str, Any]],
    *,
    reasoning_effort: str | None = None,
) -> str:
    model_name = model if model.startswith("chatgpt/") else f"chatgpt/{model}"
    return _text_from_sse_lines(
        _chatgpt_stream_lines(
            model_name,
            messages,
            reasoning_effort=reasoning_effort,
        )
    )


class ChatGPTSubscriptionShimHandler(BaseHTTPRequestHandler):
    """Tiny `/v1/chat/completions` handler for LiteLLM/OpenAI-compatible clients."""

    default_model: str | None = None

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json_response({"status": "ok", "model": self.default_model})
            return
        if self.path == "/v1/models":
            self._write_json_response(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.default_model,
                            "object": "model",
                            "owned_by": "chatgpt-subscription-shim",
                        }
                    ],
                }
            )
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404, "not found")
            return

        body_size = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(body_size))
        requested_model = str(request.get("model") or "").strip()
        model = requested_model if requested_model.startswith("chatgpt/") else self.default_model
        if not model:
            self.send_error(400, "model is required")
            return
        try:
            content = collect_chatgpt_text(
                model=str(model),
                messages=list(request.get("messages") or []),
                reasoning_effort=_reasoning_effort_from_request(request),
            )
        except Exception as exc:
            self._write_error_response(str(exc))
            return

        if request.get("stream") is True:
            self._write_streaming_response(model=str(model), content=content)
            return

        response = {
            "id": f"chatcmpl-local-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        self._write_json_response(response)

    def _write_json_response(self, response: dict[str, Any], status_code: int = 200) -> None:
        payload = json.dumps(response).encode("utf-8")
        self.send_response(status_code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _write_error_response(self, message: str) -> None:
        payload = json.dumps(
            {
                "error": {
                    "message": message[:1000],
                    "type": "chatgpt_subscription_error",
                }
            }
        ).encode("utf-8")
        self.send_response(502)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _write_streaming_response(self, *, model: str, content: str) -> None:
        response_id = f"chatcmpl-local-{int(time.time())}"
        created = int(time.time())
        chunks = [
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            },
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"content": content}, "finish_reason": None}
                ],
            },
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")


class ChatGPTSubscriptionShimServer:
    """Context manager exposing ChatGPT subscription calls as an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        default_model: str,
        install_openai_env: bool = False,
        openai_env_api_key: str = "local-shim",
    ) -> None:
        self.host = host
        self.port = port
        self.default_model = default_model
        self.install_openai_env = install_openai_env
        self.openai_env_api_key = openai_env_api_key
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._old_openai_env: dict[str, str | None] = {}

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("ChatGPT subscription shim server has not started.")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "ChatGPTSubscriptionShimServer":
        handler = type(
            "ConfiguredChatGPTSubscriptionShimHandler",
            (ChatGPTSubscriptionShimHandler,),
            {"default_model": self.default_model},
        )
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        if self.install_openai_env:
            self._old_openai_env = {
                key: os.environ.get(key)
                for key in ("OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL")
            }
            os.environ["OPENAI_API_KEY"] = self.openai_env_api_key
            os.environ["OPENAI_API_BASE"] = self.base_url
            os.environ["OPENAI_BASE_URL"] = self.base_url
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        for key, old_value in self._old_openai_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
        self._old_openai_env = {}
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
