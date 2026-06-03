# Blog 1: Distilling A 0.8B SQL Tool-Use Agent

This post uses a deterministic multi-turn SQL harness instead of a one-turn function-plan benchmark.

Dataset:

```text
birdsql/six-gym-sqlite
```

Prepared split:

```text
data/sql_agent_bird_critic/train.jsonl  # written by Notebook 01
data/sql_agent_bird_critic/eval.jsonl   # written by Notebook 01
data/sql_agent_bird_critic/dbs/         # SQLite templates
```

Current prepared data size on disk is about `554 MB`, mostly SQLite templates. The split JSONL files are small and can be committed. The SQLite templates are ignored because two files are larger than GitHub's normal file-size limit:

```text
employees_template.sqlite  # about 245 MB
airline_template.sqlite    # about 161 MB
```

On a rented GPU server, run Notebook 01 after pulling the repo. It recreates the same split and downloads the needed SQLite templates.

Teaching notebooks:

```text
1-distilling-a-0-8b-tool-calling-agent/notebooks/01_explore_sql_agent_benchmark.ipynb
1-distilling-a-0-8b-tool-calling-agent/notebooks/02_explore_teacher_sft_data.ipynb
```

Notebook 01 explores the benchmark, writes the train/eval split, and shows the harness on one task. Notebook 02 explores teacher traces and writes the final filtered SFT data. Full eval, teacher generation, and training runs use the scripts below.

Harness actions:

```json
{"action": "inspect_schema"}
{"action": "run_sql_query", "sql": "SELECT ..."}
{"action": "submit_sql", "sql": ["SQL statement 1", "SQL statement 2"]}
```

The environment executes real SQLite, returns rows/errors, and scores submitted SQL with the dataset test cases.

## BAML Clients And VSCode Extension

The harness uses BAML for OpenAI-compatible model calls. BAML does not load MLX or vLLM models by itself; it calls a running HTTP server. If the BAML VSCode extension shows an error like:

```text
connect ECONNREFUSED 127.0.0.1:8091
```

it means the selected BAML client points to a local port where no model server is running yet.

The default BAML client for `SqlAgentNextAction` is:

```text
MlxCommunityQwen35_08bMlxBf16
```

Start it before using the BAML VSCode playground or `baml-cli test`:

```bash
uv run mlx_lm.server \
  --model mlx-community/Qwen3.5-0.8B-MLX-bf16 \
  --host 127.0.0.1 \
  --port 8091 \
  --chat-template-args '{"enable_thinking": false}'
```

Then run:

```bash
uv run baml-cli test -i "SqlAgentNextAction::"
```

Configured BAML clients:

| BAML client | Model | Server to start |
| --- | --- | --- |
| `MlxCommunityQwen35_08bMlxBf16` | `mlx-community/Qwen3.5-0.8B-MLX-bf16` | `mlx_lm.server` on `8091` |
| `MlxCommunityQwen35_35bA3b8bit` | `mlx-community/Qwen3.5-35B-A3B-8bit` | `mlx_lm.server` on `8092` |
| `LiquidAiLfm25_8bA1bMlx8bit` | `LiquidAI/LFM2.5-8B-A1B-MLX-8bit` | `mlx_lm.server` on `8093` |
| `Gpt55Medium` | `gpt-5.5` | local ChatGPT shim on `8080` |
| `Qwen35_08bSqlAgentVllm` | `qwen3_5_0_8b_sql_agent` | vLLM OpenAI server on `8094` |
| `MlxCommunityQwen35_08bMlxBf16Lora` | `mlx-community/Qwen3.5-0.8B-MLX-bf16` with adapter | `mlx_lm.server --adapter-path ...` on `8095` |

The Python eval scripts can still choose a model dynamically with `--model` and `--base-url`. The BAML VSCode extension uses the static client selected in `baml_src/sql_agent.baml`, so that server must be running first.

## Serving Versus Training

Keep these two concerns separate:

```text
serving/eval/data generation:
  model server -> BAML -> SQL harness -> trace/result

training:
  canonical SFT JSONL -> trainer -> adapter/checkpoint
```

BAML lives in the serving/eval/data-generation harness. It calls a model endpoint, parses the model output into `draft` plus `output`, and the SQL environment executes the parsed action.

The trainer does not call BAML. It only sees tokenized examples built from successful harness traces.

Serving options:

| Machine | Typical server | Used for |
| --- | --- | --- |
| Mac | `mlx_lm.server` | local eval, BAML playground, teacher/student trace generation |
| NVIDIA | `vLLM` OpenAI-compatible server | GPU eval, faster batch serving, LoRA serving, future probability/logprob work |

Training options:

| Machine | Typical trainer | Used for |
| --- | --- | --- |
| Mac | MLX-Tune through `train_unsloth.py` | local Unsloth-style LoRA training |
| NVIDIA | core Unsloth through `train_unsloth.py` | later CUDA LoRA training |

For Blog 1 hard-token SFT, we train on canonical BAML decisions:

```json
{"draft": "Need schema first.", "output": {"action": "inspect_schema"}}
```

For a later soft-label/logit post, BAML is still useful for deciding the canonical target string, but BAML itself does not give token probabilities. We would score the canonical BAML target under the teacher model in teacher-forcing/scoring mode.

Teacher forcing here means: instead of asking the teacher to generate the answer freely, we feed the prompt plus the known target tokens to the model and ask, token by token, what probability it assigned to each target token. That needs a serving/scoring path that exposes logprobs, such as vLLM or a direct HF forward pass.

## NVIDIA GPU Server Safety Setup

On the RTX 4080 16GB machine we saw the system become unreachable when heavy model serving/training pushed the box too hard. The stable workaround for Blog 1 student training was:

- cap GPU power at `150W`
- keep persistence mode on
- add extra swap so host-RAM spikes do not freeze SSH immediately
- avoid local Qwen 35B teacher serving on this 16GB GPU/14GB RAM box

Run this once on a fresh Linux GPU machine:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 150

cat <<'SERVICE' | sudo tee /etc/systemd/system/nvidia-power-limit.service
[Unit]
Description=Set NVIDIA GPU power limit for safe local LLM workloads
After=multi-user.target nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -pl 150
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-power-limit.service
```

Add a 32GB swap buffer if the machine has low system RAM:

```bash
sudo fallocate -l 32G /swap-llm.img
sudo chmod 600 /swap-llm.img
sudo mkswap /swap-llm.img
sudo swapon /swap-llm.img
echo "/swap-llm.img none swap sw 0 0" | sudo tee -a /etc/fstab
```

Verify before long runs:

```bash
nvidia-smi --query-gpu=name,power.limit,power.draw,temperature.gpu,memory.used,memory.total --format=csv
free -h
swapon --show
systemctl is-active nvidia-power-limit.service
```

You do **not** need to run `nvidia-smi -pl 150` before every training run if the systemd service is active. Do verify after a reboot, driver reinstall, rented-server rebuild, or image reset. If `power.limit` is not `150.00 W`, run:

```bash
sudo systemctl restart nvidia-power-limit.service
```

The 150W cap is a stability guard, not a VRAM limiter. It reduces maximum electrical/thermal load; it does not reduce the model's GPU memory allocation. For memory pressure, use smaller models, lower context length, lower batch/concurrency, quantization, or more VRAM/system RAM.

Some rented containers do not allow changing the power limit from inside the pod. On the RunPod RTX 3090 container, `nvidia-smi -pl 150` returned `Insufficient Permissions` even as root. In that case, record the limitation and monitor the run; fixing it requires a provider/host configuration that allows power management, not a Python training-code change.

## Fresh CUDA Server Setup

For a new rented NVIDIA machine, use one repo-local environment:

```text
/workspace/small-lm-training/.venv
```

The setup script expects Python 3.11 and installs the validated CUDA stack there: Torch `2.8.0+cu128`, FlashAttention `2.8.3`, flash-linear-attention, causal-conv1d, Unsloth `2026.5.10`, TRL, and PEFT. Do not create extra throwaway envs for training.

Clone the repo on the server:

```bash
cd /workspace
git clone https://github.com/kargarisaac/small-lm-training.git
cd small-lm-training
```

Copy the existing SFT data from your Mac. This is not regenerated on the server because it came from previous teacher runs:

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 \
  outputs/gpt_5_5_medium_sql_agent_train_frozen_current.jsonl \
  root@<HOST>:/workspace/small-lm-training/outputs/
```

Then run:

```bash
1-distilling-a-0-8b-tool-calling-agent/setup_cuda_server.sh
```

The script:

- creates or reuses `.venv`
- installs the pinned CUDA student-training dependencies
- verifies CUDA, Unsloth, FlashAttention, flash-linear-attention, causal-conv1d, and xformers imports
- downloads SQLite templates directly from `birdsql/six-gym-sqlite`
- checks that the local SFT JSONL exists
- runs a dry-run tokenization/config verification

To also run a tiny actual training verification:

```bash
VERIFY_TRAIN_STEPS=2 1-distilling-a-0-8b-tool-calling-agent/setup_cuda_server.sh
```

Useful knobs:

```bash
SKIP_INSTALL=1        # reuse the existing venv packages
SKIP_DB_DOWNLOAD=1   # skip SQLite template download
SFT_DATA_URL=...     # download the SFT JSONL from an artifact URL instead of scp
```

Run the real student training explicitly:

```bash
python 1-distilling-a-0-8b-tool-calling-agent/train_unsloth.py \
  --backend cuda \
  --model unsloth/Qwen3.5-2B \
  --train-path outputs/gpt_5_5_medium_sql_agent_train_frozen_current.jsonl \
  --output-dir outputs/qwen3_5_2b_sql_agent_sft_cuda_gpt55_1046rows_4096_3epoch_lr5e-5_r32 \
  --max-seq-length 4096 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 5e-5 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --max-steps 372
```

For `LiquidAI/LFM2.5-8B-A1B` on an RTX 3090 with this Torch/Transformers stack, the default MoE `grouped_mm` backend calls a kernel that requires a newer GPU architecture. Use the Transformers eager experts backend and keep temp/Triton caches off the full root filesystem:

```bash
export TMPDIR=/dev/shm/sql-agent-tmp
export HF_HOME=/dev/shm/hf-cache-lfm
export HF_HUB_CACHE=/dev/shm/hf-cache-lfm/hub
export TRANSFORMERS_CACHE=/dev/shm/hf-cache-lfm/transformers
export TRITON_CACHE_DIR="$PWD/.triton-cache"
export UNSLOTH_COMPILE_DISABLE=1
export UNSLOTH_COMPILE_LOCATION="$PWD/.unsloth-compile-eager"
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python 1-distilling-a-0-8b-tool-calling-agent/train_unsloth.py \
  --backend cuda \
  --model LiquidAI/LFM2.5-8B-A1B \
  --train-path outputs/gpt_5_5_medium_sql_agent_train_frozen_current.jsonl \
  --output-dir outputs/lfm2_5_8b_a1b_sql_agent_sft_cuda_gpt55_1046rows_4096_3epoch_lr5e-5_r32_eager \
  --max-seq-length 4096 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 5e-5 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --max-steps 372 \
  --experts-implementation eager \
  --no-validation
```

The `--no-validation` flag is specific to this LFM/RTX-3090 pod path. The provider image has `/dev/shm` mounted `noexec`, `/` full, and the `/workspace` cache path hit a write quota during Triton validation-loss compilation. We skip in-loop validation and rely on the deterministic held-out harness eval after training.

Teacher generation is intentionally not part of this server setup. The normal flow is to generate/curate teacher traces elsewhere, copy the SFT JSONL to the GPU server, and train only the student on the rented GPU.

Validated RTX 3090 frozen-run config:

```text
teacher data: outputs/gpt_5_5_medium_sql_agent_train_frozen_current.jsonl
source rows: 1046
kept rows at 4096 tokens: 1042
train rows: 990
validation rows: 52
max sequence length: 4096
optimizer steps: 372
batch size: 1
gradient accumulation: 8
learning rate: 5e-5
LoRA rank/alpha: 32/32
```

Completed adapter runs from this config:

| Model | Runtime | Final validation loss | Local artifact |
| --- | ---: | ---: | --- |
| `unsloth/Qwen3.5-0.8B` | 956.6s | 0.3494 | `outputs/remote_training_artifacts/qwen3_5_0_8b_sql_agent_sft_cuda_gpt55_1046rows_4096_3epoch_lr5e-5_r32/` |
| `unsloth/Qwen3.5-2B` | 1327s | 0.2966 | `outputs/remote_training_artifacts/qwen3_5_2b_sql_agent_sft_cuda_gpt55_1046rows_4096_3epoch_lr5e-5_r32/` |
| `LiquidAI/LFM2.5-8B-A1B` | 3446s | n/a, `--no-validation` | `outputs/remote_training_artifacts/lfm2_5_8b_a1b_sql_agent_sft_cuda_gpt55_1046rows_4096_3epoch_lr5e-5_r32_eager_noeval/` |
| `unsloth/Qwen3.5-0.8B`, submit rows duplicated once | 1404s | 0.3591 | `outputs/remote_training_artifacts/qwen3_5_0_8b_sql_agent_sft_cuda_gpt55_1492rows_submit2x_4096_3epoch_lr5e-5_r32_rootcache/` |

On the saturated RunPod pod, the only non-destructive way to keep Qwen3.5-0.8B training on the fast path was to reuse the root compile caches from the earlier successful run. Setting `TRITON_CACHE_DIR` under `/dev/shm` failed because that mount is `noexec`; setting all Torch/Inductor caches under `/workspace` hit provider file quota; disabling compile broke the Qwen3.5 flash-linear-attention path. If this pod is rebuilt, prefer a clean executable cache path with real write quota before starting long training.

## 1. Explore And Prepare Data

Run:

```text
1-distilling-a-0-8b-tool-calling-agent/notebooks/01_explore_sql_agent_benchmark.ipynb
```

The notebook uses visible split variables:

```python
SPLIT_SEED = 42
EVAL_FRACTION = cfg.SQL_AGENT_EVAL_FRACTION
TASK_CATEGORY = "Query"
DB_FILTER = ["netflix", "movie_3", "books", "chinook"]
```

It filters to the selected task category and databases, splits each database by percentage, then writes `data/sql_agent_bird_critic/train.jsonl`, `eval.jsonl`, `stats.json`, and the SQLite templates under `data/sql_agent_bird_critic/dbs/`.

Current split written by Notebook 01:

```text
candidate rows: 1099
train rows: 879
eval rows: 220
eval fraction: 0.2 per selected database
```

## 2. Evaluate The Base Student

Model:

```text
mlx-community/Qwen3.5-0.8B-MLX-bf16
```

Command:

Start the student server:

```bash
uv run mlx_lm.server \
  --model mlx-community/Qwen3.5-0.8B-MLX-bf16 \
  --host 127.0.0.1 \
  --port 8091 \
  --chat-template-args '{"enable_thinking": false}'
```

Run the BAML harness eval:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/eval_sql_agent.py \
  --model mlx-community/Qwen3.5-0.8B-MLX-bf16 \
  --base-url http://127.0.0.1:8091/v1 \
  --data-dir data/sql_agent_bird_critic \
  --max-turns 8 \
  --max-new-tokens 1024 \
  --output outputs/qwen3_5_0_8b_mlx_sql_agent_eval.json
```

Measured result:

```text
See the final Result Summary for the current CUDA local baseline.
```

Rerun after Notebook 01 writes the percentage split.

## 3. Evaluate Teacher Baselines

### GPT 5.5 Medium

Start the local ChatGPT subscription shim:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/serve_chatgpt_shim.py \
  --model gpt-5.5 \
  --reasoning-effort medium \
  --port 8080
```

Run eval:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/eval_sql_agent.py \
  --model gpt-5.5 \
  --base-url http://127.0.0.1:8080/v1 \
  --reasoning-effort medium \
  --data-dir data/sql_agent_bird_critic \
  --max-turns 8 \
  --max-new-tokens 2048 \
  --output outputs/gpt_5_5_medium_sql_agent_eval.json
```

This uses BAML structured output: `draft` plus `output`.

Measured result:

```text
See the final Result Summary for the current 220-task eval.
```

### Qwen3.5 35B A3B 8-bit

Start MLX-LM server:

```bash
uv run mlx_lm.server \
  --model mlx-community/Qwen3.5-35B-A3B-8bit \
  --host 127.0.0.1 \
  --port 8092 \
  --chat-template-args '{"enable_thinking": false}'
```

Run eval:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/eval_sql_agent.py \
  --model mlx-community/Qwen3.5-35B-A3B-8bit \
  --base-url http://127.0.0.1:8092/v1 \
  --data-dir data/sql_agent_bird_critic \
  --max-turns 8 \
  --max-new-tokens 2048 \
  --output outputs/qwen3_5_35b_a3b_8bit_mlx_server_sql_agent_eval.json
```

This uses BAML structured output: `draft` plus `output`.

Measured result:

```text
No current 220-task result from this harness run yet.
```

### LFM2.5 8B A1B

Start MLX-LM server:

```bash
uv run mlx_lm.server \
  --model LiquidAI/LFM2.5-8B-A1B-MLX-8bit \
  --host 127.0.0.1 \
  --port 8093
```

Run eval:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/eval_sql_agent.py \
  --model LiquidAI/LFM2.5-8B-A1B-MLX-8bit \
  --base-url http://127.0.0.1:8093/v1 \
  --data-dir data/sql_agent_bird_critic \
  --max-turns 8 \
  --max-new-tokens 2048 \
  --output outputs/lfm2_5_8b_a1b_mlx_sql_agent_eval.json
```

Measured result:

```text
No current 220-task result from this harness run yet.
```

## 4. Generate BAML-Canonical Teacher Trace Rows

Current teacher for SFT:

```text
gpt-5.5 medium
```

Command:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/generate_sql_teacher_sft_rows.py \
  --model gpt-5.5 \
  --base-url http://127.0.0.1:8080/v1 \
  --reasoning-effort medium \
  --data-dir data/sql_agent_bird_critic \
  --partition train \
  --max-turns 8 \
  --max-new-tokens 2048 \
  --task-timeout-seconds 180 \
  --output outputs/gpt_5_5_medium_sql_agent_train_baml_sft_trace_rows.jsonl
```

Teacher row generation also uses the BAML structured output contract.

Teacher generation must use the current percentage split. A run over the full current train partition should report `879` attempted train tasks. If it reports `500`, it is using stale prepared data.

The script is resumable and caches after every task. For OpenAI-compatible teachers it isolates each task in a worker process by default, so a stuck subscription call becomes `teacher_runtime_error` and the run continues. The rows are BAML-canonical trace rows at this stage; Notebook 02 decides how to filter them by token length and write the final train/validation file.

## 5. Explore And Filter SFT Rows

Run:

```text
1-distilling-a-0-8b-tool-calling-agent/notebooks/02_explore_teacher_sft_data.ipynb
```

Notebook 02 converts successful BAML-canonical trace rows into the final training file:

```text
outputs/gpt_5_5_medium_sql_agent_train_frozen_20260602_213333_1046rows.jsonl
outputs/gpt_5_5_medium_sql_agent_train_frozen_20260602_213333_1046rows.report.json
outputs/gpt_5_5_medium_sql_agent_train_frozen_current.jsonl
```

The timestamped file is the immutable provenance file. The `frozen_current` file is the copy used by setup scripts and server transfers.

This matters because the harness consumes exactly one JSON action per assistant turn. The final SFT data trains on the canonical `teacher_action`, not on a BAML-canonical teacher blob that might contain extra actions.

Shared defaults:

```text
max_seq_length: 4096
batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5e-5
lora_rank: 32
lora_alpha: 32
validation_fraction: 0.05
seed: 42
```

After teacher generation finishes on the current split, Notebook 02 prints how many canonical rows fit each sequence length. Use those current numbers when choosing `--max-seq-length`.

Important Mac/CUDA difference: CUDA Unsloth applies gradient accumulation in the usual way, so `batch_size=1` and `gradient_accumulation_steps=8` means one optimizer update per eight examples. The current upstream MLX-Tune native path accepts the config field but did not forward it into MLX-LM in this environment. We patch the installed MLX-Tune package locally so `TrainingArgs` receives `grad_accumulation_steps=self.gradient_accumulation_steps`. `train_unsloth.py` checks that patch before Mac training.

One-time local patch for the current MLX-Tune version:

```bash
uv run python - <<'PY'
from pathlib import Path
import mlx_tune.sft_trainer as sft_trainer

path = Path(sft_trainer.__file__)
text = path.read_text()
old = "            adapter_file=adapter_file,\\n            grad_checkpoint=self._should_use_grad_checkpoint(),\\n        )\\n"
new = "            adapter_file=adapter_file,\\n            grad_checkpoint=self._should_use_grad_checkpoint(),\\n            grad_accumulation_steps=self.gradient_accumulation_steps,\\n        )\\n"
if new in text:
    print("MLX-Tune gradient accumulation patch already applied:", path)
elif old in text:
    path.write_text(text.replace(old, new, 1))
    print("Patched MLX-Tune gradient accumulation:", path)
else:
    raise SystemExit(f"Could not find expected TrainingArgs block in {path}")
PY
```

Current frozen teacher data used for the CUDA runs:

```text
teacher train success: 446/879 = 50.7%
source rows: 1046
4096 tokens: 1042/1046 rows fit
4096-token train/validation split: 990/52
p50 tokens: 1786
p90 tokens: 2948
p95 tokens: 3208
max tokens: 15836
```

For MLX-LM, training iterations are microsteps; gradient accumulation changes optimizer-update frequency but does not reduce how many examples must run through forward/backward for one pass. On this Mac, full `3072` context can still hit the Metal allocation limit during backward. CUDA Unsloth can use the larger effective batch through `--grad-accum 8`.

## 6. Train The Student

### Mac Path: MLX-Tune

On Apple Silicon, `train_unsloth.py` imports `mlx_tune` and keeps the training code shaped like Unsloth:

```bash
uv pip install mlx-tune

uv run python 1-distilling-a-0-8b-tool-calling-agent/train_unsloth.py \
  --train-path outputs/gpt_5_5_medium_sql_agent_train_frozen_current.jsonl \
  --output-dir outputs/qwen3_5_0_8b_mlx_tune_sql_agent_gpt_teacher_sft_2560_gradaccum1 \
  --max-seq-length 2560 \
  --grad-accum 1 \
  --learning-rate 5e-5
```

The script uses response-only training, so the loss is on the assistant JSON decision rather than on the whole prompt.

The same file can still run on CUDA later by forcing the CUDA backend:

```bash
uv pip install unsloth

uv run python 1-distilling-a-0-8b-tool-calling-agent/train_unsloth.py \
  --backend cuda \
  --model unsloth/Qwen3.5-2B \
  --train-path outputs/gpt_5_5_medium_sql_agent_train_frozen_current.jsonl \
  --output-dir outputs/qwen3_5_2b_sql_agent_sft_cuda_gpt55_1046rows_4096_3epoch_lr5e-5_r32 \
  --max-seq-length 4096 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 5e-5 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --max-steps 372
```

## 7. Evaluate The Tuned Student

The evaluation harness must use the same prompt, parser, tools, and deterministic SQL scoring. There are two supported paths:

- `eval_sql_agent.py`: OpenAI-compatible HTTP model server, used for GPT/MLX/vLLM-style serving.
- `eval_sql_agent_local.py`: local HF/PEFT model in-process, used for CUDA base and adapter evals without starting a separate server.

The completed CUDA evals used `eval_sql_agent_local.py`. It renders the same BAML prompt/messages, then runs the same SQL harness and parser locally.

On this Mac, serve the MLX-Tune adapter with `mlx_lm.server`:

```bash
uv run mlx_lm.server \
  --model mlx-community/Qwen3.5-0.8B-MLX-bf16 \
  --adapter-path outputs/qwen3_5_0_8b_mlx_tune_sql_agent_gpt_teacher_sft_2560_gradaccum1/adapter \
  --host 127.0.0.1 \
  --port 8095 \
  --chat-template-args '{"enable_thinking": false}'
```

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/eval_sql_agent.py \
  --model mlx-community/Qwen3.5-0.8B-MLX-bf16 \
  --base-url http://127.0.0.1:8095/v1 \
  --data-dir data/sql_agent_bird_critic \
  --max-turns 8 \
  --max-new-tokens 512 \
  --task-timeout-seconds 180 \
  --output outputs/qwen3_5_0_8b_mlx_tune_sql_agent_gpt_teacher_sft_2560_gradaccum1_eval.json
```

On NVIDIA, evaluate the saved HF/PEFT adapter directly:

```bash
python 1-distilling-a-0-8b-tool-calling-agent/eval_sql_agent_local.py \
  --model unsloth/Qwen3.5-2B \
  --adapter-path outputs/qwen3_5_2b_sql_agent_sft_cuda_gpt55_1046rows_4096_3epoch_lr5e-5_r32/adapter \
  --data-dir data/sql_agent_bird_critic \
  --partition eval \
  --max-seq-length 8192 \
  --max-new-tokens 512 \
  --max-turns 8 \
  --task-timeout-seconds 180 \
  --temperature 0.0 \
  --dtype bf16 \
  --output outputs/qwen3_5_2b_sql_agent_sft_cuda_gpt55_1046rows_4096_3epoch_lr5e-5_r32_eval.json
```

Use the same command without `--adapter-path` for the non-finetuned baseline.

vLLM serving is still the right path when we need OpenAI-compatible serving, LoRA hot-loading, batching, or logprobs for a later soft-label post:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model unsloth/Qwen3.5-2B \
  --enable-lora \
  --lora-modules sql_agent=outputs/qwen3_5_2b_sql_agent_sft_cuda_gpt55_1046rows_4096_3epoch_lr5e-5_r32/adapter \
  --served-model-name qwen3_5_2b_sql_agent \
  --port 8094
```

## Result Summary

Current frozen-run eval split: `220` tasks from `data/sql_agent_bird_critic/eval.jsonl`.

| Run | Success | Submitted | Parse failures | Repeated-action failures | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| GPT 5.5 medium teacher | 115/220 = 52.3% | 220 | 0 | 0 | OpenAI-compatible ChatGPT shim, reasoning effort `medium` |
| Qwen3.5-0.8B base | 1/220 = 0.5% | 1 | 11 | 208 | `unsloth/Qwen3.5-0.8B`, no adapter |
| Qwen3.5-0.8B SFT | 44/220 = 20.0% | 204 | 12 | 4 | 653-row GPT teacher data, 3072-token filter |
| Qwen3.5-0.8B SFT | 44/220 = 20.0% | 204 | 6 | 10 | 1046-row GPT teacher data, 4096-token filter, r32 |
| Qwen3.5-0.8B SFT | 38/220 = 17.3% | 199 | 9 | 11 | 1046-row data with `submit_sql` rows duplicated once; negative result |
| Qwen3.5-2B base | 0/220 = 0.0% | 1 | 0 | 219 | `unsloth/Qwen3.5-2B`, no adapter |
| Qwen3.5-2B SFT | 55/220 = 25.0% | 189 | 9 | 22 | 653-row GPT teacher data, 3072-token filter, r16 |
| Qwen3.5-2B SFT | 62/220 = 28.2% | 196 | 13 | 11 | 653-row GPT teacher data, 4096-token filter, r32 |
| Qwen3.5-2B SFT | 57/220 = 25.9% | 206 | 5 | 9 | 1046-row GPT teacher data, 4096-token filter, r32 |
| LFM2.5-8B-A1B SFT | 47/220 = 21.4% | 186 | 7 | 27 | 1046-row GPT teacher data, 4096-token filter, r32, eager MoE experts, no in-loop validation |

The main measured effect is harness control: base Qwen students mostly repeat actions and almost never submit. SFT teaches them to interact with the harness and submit, but SQL correctness still trails the GPT teacher by a wide margin. The 1046-row Qwen3.5-0.8B run improved validation loss and submission rate, but not held-out task success. Duplicating `submit_sql` rows was a clean class-weighting experiment, but it made the 0.8B model worse: it lost 17 previously-correct tasks, gained 11 new correct tasks, and increased SQL execution errors. LFM2.5-8B-A1B also did not beat the smaller Qwen3.5-2B run on this harness.

Ignored local artifacts:

```text
outputs/gpt_5_5_medium_sql_agent_eval_220_timeout180.json
outputs/remote_eval_results/qwen3_5_0_8b_base_eval_full_ctx8192_maxnew512_timeout180.json
outputs/remote_eval_results/qwen3_5_0_8b_sql_agent_sft_cuda_gpt55_653rows_3072_2epoch_eval_full_ctx8192_maxnew512_timeout180_pyfix.json
outputs/remote_eval_results/qwen3_5_0_8b_sql_agent_sft_cuda_gpt55_1046rows_4096_3epoch_lr5e-5_r32_eval_full_ctx8192_maxnew512_timeout180.json
outputs/remote_eval_results/qwen3_5_0_8b_sql_agent_sft_cuda_gpt55_1492rows_submit2x_4096_3epoch_lr5e-5_r32_rootcache_eval_full_ctx8192_maxnew512_timeout180.json
outputs/remote_eval_results/lfm2_5_8b_a1b_sql_agent_sft_cuda_gpt55_1046rows_4096_3epoch_lr5e-5_r32_eager_noeval_eval_full_ctx8192_maxnew512_timeout180.json
outputs/remote_eval_results/qwen3_5_2b_base_eval_full_ctx8192_maxnew512_timeout180.json
outputs/remote_eval_results/qwen3_5_2b_sql_agent_sft_cuda_gpt55_653rows_3072_2epoch_eval_full_ctx8192_maxnew512_timeout180.json
outputs/remote_eval_results/qwen3_5_2b_sql_agent_sft_cuda_gpt55_653rows_4096_3epoch_lr5e-5_r32_eval_full_ctx8192_maxnew512_timeout180.json
outputs/remote_eval_results/qwen3_5_2b_sql_agent_sft_cuda_gpt55_1046rows_4096_3epoch_lr5e-5_r32_eval_full_ctx8192_maxnew512_timeout180_tmpdirdevshm.json
outputs/remote_training_artifacts/qwen3_5_0_8b_sql_agent_sft_cuda_gpt55_653rows_3072_2epoch_lr5e-5_r16/
outputs/remote_training_artifacts/qwen3_5_0_8b_sql_agent_sft_cuda_gpt55_1046rows_4096_3epoch_lr5e-5_r32/
outputs/remote_training_artifacts/qwen3_5_0_8b_sql_agent_sft_cuda_gpt55_1492rows_submit2x_4096_3epoch_lr5e-5_r32_rootcache/
outputs/remote_training_artifacts/qwen3_5_2b_sql_agent_sft_cuda_gpt55_653rows_3072_2epoch_lr5e-5_r16/
outputs/remote_training_artifacts/qwen3_5_2b_sql_agent_sft_cuda_gpt55_653rows_4096_3epoch_lr5e-5_r32/
outputs/remote_training_artifacts/qwen3_5_2b_sql_agent_sft_cuda_gpt55_1046rows_4096_3epoch_lr5e-5_r32/
outputs/remote_training_artifacts/lfm2_5_8b_a1b_sql_agent_sft_cuda_gpt55_1046rows_4096_3epoch_lr5e-5_r32_eager_noeval/
```

## Next Fixes

- Improve teacher trace quality and coverage before increasing model size again.
- Keep enough checkpoints, or save the best checkpoint explicitly, before relying on validation loss for early stopping. The submit-weighted run's best validation checkpoint was deleted by `save_total_limit=2`.
- Inspect remaining failed traces to separate SQL-reasoning misses from harness-control misses before changing the model or LoRA recipe again.
- Avoid more action-level reweighting until failure analysis shows it addresses correctness rather than just shifting tool-call frequencies.
