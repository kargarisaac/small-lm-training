#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
SFT_PATH="${SFT_PATH:-outputs/gpt_5_5_medium_sql_agent_train_frozen_current.jsonl}"
VERIFY_MAX_SEQ_LENGTH="${VERIFY_MAX_SEQ_LENGTH:-3072}"
VERIFY_LIMIT="${VERIFY_LIMIT:-16}"
VERIFY_TRAIN_STEPS="${VERIFY_TRAIN_STEPS:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_DB_DOWNLOAD="${SKIP_DB_DOWNLOAD:-0}"
SFT_DATA_URL="${SFT_DATA_URL:-}"

echo "Repo: $REPO_ROOT"
echo "Python: $($PYTHON_BIN --version)"

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit("This setup expects Python 3.11 because the validated FlashAttention wheel is cp311.")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
  echo "ERROR: nvidia-smi not found. This setup script expects an NVIDIA CUDA server." >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [[ "$SKIP_INSTALL" != "1" ]]; then
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache}"
  mkdir -p "$PIP_CACHE_DIR"

  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.8.0+cu128" \
    "torchvision==0.23.0+cu128"
  python -m pip install \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
  python -m pip install --no-deps \
    "unsloth==2026.5.10" \
    "unsloth_zoo==2026.5.5"
  python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "transformers==5.5.0" \
    "datasets==3.6.0" \
    "accelerate==1.13.0" \
    "peft==0.19.1" \
    "trl==0.24.0" \
    "baml-py==0.222.0" \
    "httpx>=0.28.1" \
    "sentencepiece" \
    "protobuf" \
    "safetensors==0.8.0rc1" \
    "tokenizers==0.22.2" \
    "huggingface_hub==1.17.0" \
    "hf-transfer" \
    "cut-cross-entropy==25.1.1" \
    "msgspec==0.21.1" \
    "pillow==12.2.0" \
    "torchao==0.17.0+cu128" \
    "tyro==1.0.13" \
    "bitsandbytes==0.49.2" \
    "diffusers==0.38.0" \
    "nest-asyncio" \
    "pydantic==2.13.4" \
    "xformers==0.0.32.post2"
  python -m pip install --no-build-isolation "causal-conv1d==1.6.2.post1"
  python -m pip install "flash-linear-attention==0.5.0"
  python -m pip check
fi

mkdir -p "$(dirname "$SFT_PATH")"
if [[ ! -f "$SFT_PATH" && -n "$SFT_DATA_URL" ]]; then
  python - <<PY
from pathlib import Path
from urllib.request import urlretrieve

target = Path("$SFT_PATH")
target.parent.mkdir(parents=True, exist_ok=True)
urlretrieve("$SFT_DATA_URL", target)
print(f"Downloaded SFT data to {target}")
PY
fi

if [[ ! -f "$SFT_PATH" ]]; then
  cat >&2 <<MSG
ERROR: missing SFT data: $SFT_PATH

This file is generated from previous teacher runs and is intentionally not committed.
Copy it from your Mac, for example:

scp -P <PORT> -i ~/.ssh/id_ed25519 \\
  outputs/gpt_5_5_medium_sql_agent_train_frozen_current.jsonl \\
  root@<HOST>:$REPO_ROOT/outputs/

Or rerun this script with SFT_DATA_URL=<download-url> if you publish the JSONL as an artifact.
MSG
  exit 1
fi

if [[ "$SKIP_DB_DOWNLOAD" != "1" ]]; then
  python - <<'PY'
from common import config as cfg
from common import sql_agent

data_dir = cfg.DATA_DIR / "sql_agent_bird_critic"
rows = sql_agent.load_rows(data_dir, "train") + sql_agent.load_rows(data_dir, "eval")
db_ids = {row["db_id"] for row in rows}
print(f"Downloading/verifying {len(db_ids)} SQLite templates from {cfg.SQL_AGENT_DATASET}...")
sql_agent.copy_database_templates(data_dir, db_ids)
print(f"SQLite templates ready under {data_dir / 'dbs'}")
PY
fi

python - <<'PY'
import importlib.metadata as metadata
import importlib.util
import torch
import transformers
import datasets

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
for package_name in ["unsloth", "unsloth_zoo", "flash-attn", "flash-linear-attention", "causal-conv1d"]:
    print(f"{package_name}:", metadata.version(package_name))
for module_name in ["unsloth", "flash_attn", "fla", "causal_conv1d", "causal_conv1d_cuda", "xformers"]:
    print(f"{module_name} importable:", importlib.util.find_spec(module_name) is not None)
PY

python 1-distilling-a-0-8b-tool-calling-agent/train_unsloth.py \
  --backend cuda \
  --train-path "$SFT_PATH" \
  --output-dir outputs/cuda_setup_verify \
  --limit "$VERIFY_LIMIT" \
  --max-seq-length "$VERIFY_MAX_SEQ_LENGTH" \
  --max-steps 1 \
  --dry-run

if [[ "$VERIFY_TRAIN_STEPS" != "0" ]]; then
  python 1-distilling-a-0-8b-tool-calling-agent/train_unsloth.py \
    --backend cuda \
    --train-path "$SFT_PATH" \
    --output-dir outputs/cuda_setup_verify \
    --limit "$VERIFY_LIMIT" \
    --max-seq-length "$VERIFY_MAX_SEQ_LENGTH" \
    --max-steps "$VERIFY_TRAIN_STEPS"
fi

echo
echo "CUDA server setup is ready."
echo "SFT data: $SFT_PATH"
echo "Benchmark data: data/sql_agent_bird_critic"
echo
echo "Next real training command:"
echo "python 1-distilling-a-0-8b-tool-calling-agent/train_unsloth.py \\"
echo "  --backend cuda \\"
echo "  --train-path $SFT_PATH \\"
echo "  --output-dir outputs/qwen3_5_0_8b_sql_agent_sft_cuda_gpt55_653rows_3072_2epoch_lr5e-5_r16 \\"
echo "  --max-seq-length 3072 \\"
echo "  --batch-size 1 \\"
echo "  --grad-accum 8 \\"
echo "  --learning-rate 5e-5 \\"
echo "  --lora-rank 16 \\"
echo "  --lora-alpha 16 \\"
echo "  --max-steps 146"
