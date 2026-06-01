from __future__ import annotations

from typing import Any, Callable
import json
import os

from . import config as cfg


Generator = Callable[[list[dict[str, str]]], str]


def make_baml_generator(
    *,
    model_name: str,
    max_new_tokens: int = 128,
    base_url: str,
    api_key_env: str | None = None,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
) -> Generator:
    from baml_py import ClientRegistry
    from baml_client import b
    from baml_client.types import SqlAgentMessage

    os.environ.setdefault("BAML_LOG", "warn")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    client_options: dict[str, Any] = {
        "base_url": base_url.rstrip("/"),
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }
    if api_key:
        client_options["api_key"] = api_key
    if "qwen" in model_name.lower():
        client_options["chat_template_kwargs"] = {"enable_thinking": cfg.QWEN_ENABLE_THINKING}
    if reasoning_effort:
        client_options["reasoning_effort"] = reasoning_effort

    client_registry = ClientRegistry()
    client_registry.add_llm_client("SqlAgentRuntimeClient", "openai-generic", client_options)
    client_registry.set_primary("SqlAgentRuntimeClient")

    def generate(messages: list[dict[str, str]]) -> str:
        decision = b.with_options(client_registry=client_registry).SqlAgentNextAction(
            [SqlAgentMessage(role=message["role"], content=message["content"]) for message in messages],
        )
        return json.dumps(
            {"draft": decision.draft, "output": decision.output.model_dump()},
            ensure_ascii=False,
        )

    return generate
