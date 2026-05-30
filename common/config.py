from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any
import json
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

NESTFUL_DATASET = "ibm-research/nestful"
NESTFUL_SPLIT_SEED = 42
NESTFUL_EVAL_FRACTION = 0.2
HF_STUDENT_MODEL = "Qwen/Qwen3.5-0.8B"
MLX_STUDENT_MODEL = "mlx-community/Qwen3.5-0.8B-MLX-bf16"
UNSLOTH_STUDENT_MODEL = "Qwen/Qwen3.5-0.8B"
QWEN_TEACHER_MODEL = "mlx-community/Qwen3.5-35B-A3B-8bit"
LFM_TEACHER_MODEL = "LiquidAI/LFM2.5-8B-A1B-MLX-8bit"
GPT_TEACHER_MODEL = "gpt-5.5"
GPT_TEACHER_REASONING_EFFORT = "medium"


def filename_slug(value: str) -> str:
    return "_".join(part for part in "".join(ch if ch.isalnum() else "_" for ch in value).split("_") if part).lower()


def load_dotenv_if_present() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]
    return repr(value)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_safe(value), indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(make_json_safe(row), ensure_ascii=False) for row in rows)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")
