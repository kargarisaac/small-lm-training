from __future__ import annotations

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import uuid4
import json
import os
import time
import warnings

import httpx
import litellm
from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import ensure_chatgpt_session_id, get_chatgpt_default_headers


os.environ.setdefault("LITELLM_LOG", "ERROR")
warnings.filterwarnings("ignore", message="Pydantic serializer warnings:*")
litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.turn_off_message_logging = True

TOOL_INSTRUCTIONS = """You are a function-calling model inside an evaluation harness.
Return exactly one JSON object and no extra text:
{"function":{"name":"tool_name","arguments":{"arg":"value"}}}
Use only the available tools. Include required arguments. Omit arguments that are not needed.
"""


def reasoning_effort_from_request(request: dict[str, Any]) -> str | None:
    reasoning = request.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        return str(reasoning["effort"])
    effort = request.get("reasoning_effort")
    return str(effort) if effort else None


def response_messages(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    instructions: list[str] = []
    user_messages: list[dict[str, str]] = []
    if tools:
        instructions.append(TOOL_INSTRUCTIONS)
        instructions.append("Available tools:\n" + json.dumps(tools, ensure_ascii=False))
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(str(item.get("text") or item.get("content") or item) if isinstance(item, dict) else str(item) for item in content)
        if role in {"system", "developer"}:
            instructions.append(str(content))
        elif role in {"user", "assistant"}:
            user_messages.append({"role": role, "content": str(content)})
    return "\n\n".join(part for part in instructions if part), user_messages or [{"role": "user", "content": ""}]


def stream_chatgpt_lines(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    reasoning_effort: str | None,
) -> Iterator[str]:
    authenticator = Authenticator()
    token = authenticator.get_access_token()
    api_base = authenticator.get_api_base().rstrip("/")
    instructions, response_input = response_messages(messages, tools)
    body: dict[str, Any] = {
        "model": model.removeprefix("chatgpt/"),
        "instructions": instructions,
        "input": response_input,
        "stream": True,
        "store": False,
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    headers = get_chatgpt_default_headers(
        token,
        authenticator.get_account_id(),
        ensure_chatgpt_session_id({"litellm_call_id": f"distillation-blogs-{uuid4()}"}),
    )
    with httpx.stream("POST", f"{api_base}/responses", headers=headers, json=body, timeout=300) as response:
        if response.status_code >= 400:
            detail = response.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"ChatGPT subscription request failed: {response.status_code} {detail}")
        yield from response.iter_lines()


def text_from_sse(lines: Iterator[str]) -> str:
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


class Handler(BaseHTTPRequestHandler):
    default_model = "gpt-5.5"
    default_reasoning_effort = "medium"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            self.write_json({"status": "ok", "model": self.default_model, "reasoning_effort": self.default_reasoning_effort})
        elif self.path == "/v1/models":
            self.write_json({"object": "list", "data": [{"id": self.default_model, "object": "model"}]})
        else:
            self.send_error(404, "not found")

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404, "not found")
            return
        request = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))))
        model = str(request.get("model") or self.default_model)
        reasoning_effort = reasoning_effort_from_request(request) or self.default_reasoning_effort
        try:
            content = text_from_sse(
                stream_chatgpt_lines(
                    model,
                    list(request.get("messages") or []),
                    list(request.get("tools") or []),
                    reasoning_effort,
                )
            )
        except Exception as exc:
            self.write_json({"error": {"message": str(exc)[:1000], "type": "chatgpt_subscription_error"}}, status=500)
            return
        self.write_json(
            {
                "id": f"chatcmpl-local-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            }
        )

    def write_json(self, value: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve_chatgpt_subscription(model: str, host: str, port: int, reasoning_effort: str) -> ThreadingHTTPServer:
    Handler.default_model = model
    Handler.default_reasoning_effort = reasoning_effort
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"ChatGPT subscription shim serving {model} ({reasoning_effort}) at http://{host}:{port}/v1")
    server.serve_forever()
    return server
