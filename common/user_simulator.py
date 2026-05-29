from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os

from .config import _configure_litellm_for_notebooks, required_env


__all__ = [
    "TauBenchUserSimulatorRuntime",
    "public_user_simulator_args",
    "start_tau_bench_user_simulator_from_env",
]


@dataclass
class TauBenchUserSimulatorRuntime:
    model: str
    args: dict[str, Any]
    shim: Any
    shim_model: str


def _start_chatgpt_subscription_shim(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    default_model: str,
    api_key: str = "local-shim",
) -> Any:
    """Start the local OpenAI-compatible ChatGPT subscription shim."""

    if not default_model.strip():
        raise RuntimeError("ChatGPT subscription shim default_model cannot be empty.")
    _configure_litellm_for_notebooks()
    from .chatgpt_subscription_shim import ChatGPTSubscriptionShimServer  # noqa: PLC0415

    shim = ChatGPTSubscriptionShimServer(
        host=host,
        port=port,
        default_model=default_model,
        install_openai_env=True,
        openai_env_api_key=api_key,
    )
    shim.__enter__()
    return shim


def _chatgpt_subscription_model_from_user_simulator(user_simulator_model: str) -> str:
    if not user_simulator_model.startswith("openai/"):
        raise RuntimeError(
            "The local ChatGPT subscription user simulator expects TAU_BENCH_USER_SIMULATOR_LLM "
            "to use an openai/... LiteLLM model name."
        )
    model = user_simulator_model.split("/", 1)[1].strip()
    if not model:
        raise RuntimeError("TAU_BENCH_USER_SIMULATOR_LLM has an empty model name after openai/.")
    return model


def _chatgpt_subscription_user_simulator_args(
    *,
    base_url: str,
    api_key: str = "local-shim",
    temperature: float = 0.0,
    num_retries: int = 6,
    timeout: int = 300,
) -> dict[str, Any]:
    return {
        "api_base": base_url,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
        "num_retries": num_retries,
        "timeout": timeout,
    }


def public_user_simulator_args(args: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in args.items() if key != "api_key"}


def start_tau_bench_user_simulator_from_env(
    *,
    existing_shim: Any | None = None,
) -> TauBenchUserSimulatorRuntime:
    if existing_shim is not None:
        existing_shim.__exit__(None, None, None)

    model = required_env("TAU_BENCH_USER_SIMULATOR_LLM")
    backend = os.getenv("TAU_BENCH_USER_SIMULATOR_BACKEND", "chatgpt_subscription").strip()
    if backend == "litellm":
        return TauBenchUserSimulatorRuntime(
            model=model,
            args={
                "temperature": 0.0,
                "num_retries": 6,
                "timeout": 300,
            },
            shim=None,
            shim_model=model,
        )
    if backend != "chatgpt_subscription":
        raise RuntimeError(
            "TAU_BENCH_USER_SIMULATOR_BACKEND must be 'chatgpt_subscription' or 'litellm'."
        )

    shim_model = _chatgpt_subscription_model_from_user_simulator(model)
    shim_api_key = os.getenv("TAU_BENCH_CHATGPT_SHIM_API_KEY", "local-shim")
    shim = _start_chatgpt_subscription_shim(
        port=int(os.getenv("TAU_BENCH_CHATGPT_SHIM_PORT", "0")),
        default_model=shim_model,
        api_key=shim_api_key,
    )
    args = _chatgpt_subscription_user_simulator_args(
        base_url=shim.base_url,
        api_key=shim_api_key,
        temperature=0.0,
        num_retries=6,
        timeout=300,
    )
    return TauBenchUserSimulatorRuntime(
        model=model,
        args=args,
        shim=shim,
        shim_model=shim_model,
    )
