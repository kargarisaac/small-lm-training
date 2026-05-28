from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
import json
import logging
import os
import sys


COMMON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = COMMON_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

STUDENT_MODEL = "mlx-community/Qwen3.5-0.8B-MLX-bf16"
TEACHER_MODEL = "mlx-community/Qwen3.5-35B-A3B-8bit"
TOKENIZER_MODEL = STUDENT_MODEL
TAU_BENCH_REPO_REVISION = "c42db6cc223ef37c02ef2fb2f605ae0a4ca9afd6"
MAX_NEW_TOKENS = 2048
_GENERATION_STOP_STRINGS = ["<|im_end|>", "<|endoftext|>"]

__all__ = [
    "MAX_NEW_TOKENS",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "STUDENT_MODEL",
    "TAU_BENCH_REPO_REVISION",
    "TEACHER_MODEL",
    "TOKENIZER_MODEL",
    "MlflowConfig",
    "NotebookPaths",
    "TauBenchRetailEvalConfig",
    "TeacherConfig",
    "env_flag",
    "filename_slug",
    "load_jsonl",
    "load_tokenizer",
    "make_json_safe",
    "percentile_int",
    "required_env",
    "setup_notebook_paths",
    "teacher_config_from_env",
    "write_jsonl",
]


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required environment variable {name}. Set it in .env before running.")
    return value.strip()


def _configure_litellm_for_notebooks() -> None:
    """Keep LiteLLM provider errors out of notebook stdout."""

    os.environ.setdefault("LITELLM_LOG", "ERROR")
    try:
        import litellm  # noqa: PLC0415

        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        litellm.turn_off_message_logging = True
    except Exception:
        return


@contextmanager
def _quiet_external_logs(enabled: bool = True):
    if not enabled:
        yield
        return

    _configure_litellm_for_notebooks()
    loguru_logger = None
    disabled_loguru_modules: list[str] = []
    try:
        from loguru import logger as imported_loguru_logger  # noqa: PLC0415

        loguru_logger = imported_loguru_logger
        for module_name in ("tau2", "litellm", "httpx", "httpcore", "urllib3"):
            loguru_logger.disable(module_name)
            disabled_loguru_modules.append(module_name)
    except Exception:
        loguru_logger = None

    stdlib_logger_states: dict[str, tuple[bool, int, bool]] = {}
    for logger_name in ("tau2", "litellm", "httpx", "httpcore", "urllib3"):
        stdlib_logger = logging.getLogger(logger_name)
        stdlib_logger_states[logger_name] = (
            stdlib_logger.disabled,
            stdlib_logger.level,
            stdlib_logger.propagate,
        )
        stdlib_logger.disabled = True

    try:
        yield
    finally:
        for logger_name, (disabled, level, propagate) in stdlib_logger_states.items():
            stdlib_logger = logging.getLogger(logger_name)
            stdlib_logger.disabled = disabled
            stdlib_logger.setLevel(level)
            stdlib_logger.propagate = propagate

        if loguru_logger is not None:
            for module_name in disabled_loguru_modules:
                loguru_logger.enable(module_name)


def _default_teacher_server_base_url(provider: str | None = None) -> str:
    provider = provider or os.getenv("TEACHER_PROVIDER", "mlx_raw_server")
    if provider == "ollama_raw":
        return "http://127.0.0.1:11434"
    if provider == "vllm_raw":
        return "http://127.0.0.1:8092"
    if provider == "chatgpt_raw":
        return "http://127.0.0.1:8080/v1"
    return "http://127.0.0.1:8080"


def percentile_int(values: list[int], fraction: float) -> int:
    if not values:
        raise ValueError("percentile_int needs at least one value.")
    sorted_values = sorted(values)
    return sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * fraction))]


@dataclass
class TeacherConfig:
    provider: str = os.getenv("TEACHER_PROVIDER", "mlx_raw_server")
    server_base_url: str | None = None
    model_name: str = os.getenv("TEACHER_MODEL", TEACHER_MODEL)
    request_model: str = os.getenv("TEACHER_REQUEST_MODEL", "default_model")
    enable_thinking: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    reasoning_effort: str | None = os.getenv("TEACHER_REASONING_EFFORT")
    request_timeout_seconds: int = int(os.getenv("TEACHER_REQUEST_TIMEOUT_SECONDS", "180"))
    max_new_tokens: int = int(os.getenv("TEACHER_MAX_NEW_TOKENS", str(MAX_NEW_TOKENS)))

    def __post_init__(self) -> None:
        if self.server_base_url is None:
            self.server_base_url = os.getenv(
                "TEACHER_SERVER_BASE_URL",
                _default_teacher_server_base_url(self.provider),
            )
        self.server_base_url = self.server_base_url.rstrip("/")


def teacher_config_from_env(
    *,
    default_provider: str = "vllm_raw",
    default_model: str = TEACHER_MODEL,
) -> TeacherConfig:
    provider = os.getenv("TEACHER_PROVIDER", default_provider)
    model_name = os.getenv("TEACHER_MODEL", default_model)
    return TeacherConfig(
        provider=provider,
        server_base_url=os.getenv("TEACHER_SERVER_BASE_URL", _default_teacher_server_base_url(provider)).rstrip("/"),
        model_name=model_name,
        request_model=os.getenv("TEACHER_REQUEST_MODEL", model_name),
        temperature=float(os.getenv("TEACHER_TEMPERATURE", "0.0")),
        top_p=float(os.getenv("TEACHER_TOP_P", "1.0")),
        top_k=int(os.getenv("TEACHER_TOP_K", "0")),
        reasoning_effort=os.getenv("TEACHER_REASONING_EFFORT"),
        request_timeout_seconds=int(os.getenv("TEACHER_REQUEST_TIMEOUT_SECONDS", "600")),
        max_new_tokens=int(os.getenv("TEACHER_MAX_NEW_TOKENS", str(MAX_NEW_TOKENS))),
    )


@dataclass(frozen=True)
class MlflowConfig:
    enabled: bool = env_flag("TAU_BENCH_MLFLOW_ENABLED", False)
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
    experiment_name: str = os.getenv("TAU_BENCH_MLFLOW_EXPERIMENT_NAME", "distillation-blogs-tau3")
    log_full_artifacts: bool = env_flag("TAU_BENCH_MLFLOW_LOG_FULL_ARTIFACTS", False)
    log_spans: bool = env_flag("TAU_BENCH_MLFLOW_LOG_SPANS", False)


@dataclass(frozen=True)
class NotebookPaths:
    root: Path
    blog_dir: Path
    data_dir: Path
    output_dir: Path
    env_path: Path
    dotenv_loaded: bool


@dataclass(frozen=True)
class TauBenchRetailEvalConfig:
    dataset_revision: str
    student_model_name: str
    user_simulator_model: str
    user_simulator_args: dict[str, Any] | None = None
    max_steps: int = 100
    max_errors: int = 10
    max_new_tokens: int = MAX_NEW_TOKENS
    seed: int = 42
    pass_at: int = 1
    model_role: str = "student"


def setup_notebook_paths(
    *,
    blog_dir_name: str | None = None,
    load_env: bool = True,
) -> NotebookPaths:
    cwd = Path.cwd().resolve()
    if (cwd / "common" / "config.py").exists():
        root = cwd
    elif (cwd.parent / "common" / "config.py").exists():
        root = cwd.parent
    else:
        raise RuntimeError("Run this notebook from the repo root or from a blog folder under the repo root.")

    if blog_dir_name is not None:
        blog_dir = root / blog_dir_name
    elif cwd != root:
        blog_dir = cwd
    else:
        blog_dir = root

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    env_path = root / ".env"
    dotenv_loaded = False
    if load_env:
        from dotenv import load_dotenv  # noqa: PLC0415

        dotenv_loaded = load_dotenv(env_path, override=False)

    return NotebookPaths(
        root=root,
        blog_dir=blog_dir,
        data_dir=root / "data",
        output_dir=root / "outputs",
        env_path=env_path,
        dotenv_loaded=dotenv_loaded,
    )


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(make_json_safe(row)) for row in rows)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def make_json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return make_json_safe(value.model_dump(mode="json"))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    return repr(value)


def load_tokenizer(model_name: str = TOKENIZER_MODEL):
    from transformers import AutoTokenizer  # noqa: PLC0415

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _teacher_config_to_dict(config: TeacherConfig) -> dict[str, Any]:
    return {
        "provider": config.provider,
        "server_base_url": config.server_base_url,
        "model_name": config.model_name,
        "request_model": config.request_model,
        "enable_thinking": config.enable_thinking,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "reasoning_effort": config.reasoning_effort,
        "request_timeout_seconds": config.request_timeout_seconds,
        "max_new_tokens": config.max_new_tokens,
    }


def filename_slug(value: str) -> str:
    return "_".join(
        part
        for part in "".join(ch if ch.isalnum() else "_" for ch in value).split("_")
        if part
    ).lower()
