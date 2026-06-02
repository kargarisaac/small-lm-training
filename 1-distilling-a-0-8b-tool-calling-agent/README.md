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
No current BAML-harness result is checked in yet.
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
No current BAML-harness result is checked in yet.
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
No current BAML-harness result is checked in yet.
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
No current BAML-harness result is checked in yet.
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
outputs/gpt_5_5_medium_sql_agent_train_sft_canonical_3072.jsonl
outputs/gpt_5_5_medium_sql_agent_train_sft_canonical_3072.report.json
```

This matters because the harness consumes exactly one JSON action per assistant turn. The final SFT data trains on the canonical `teacher_action`, not on a BAML-canonical teacher blob that might contain extra actions.

Shared defaults:

```text
max_seq_length: 3072
batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1e-5
lora_rank: 16
lora_alpha: 16
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

Current patched Mac dry-run coverage from the canonical file on this machine. For MLX-LM, training iterations are microsteps; gradient accumulation changes optimizer-update frequency but does not reduce how many examples must run through forward/backward for one pass.

```text
2048 tokens: 471/721 rows fit
2560 tokens: 615/721 rows fit
3072 tokens: 721/721 rows fit
```

On this Mac, full `3072` context still hits the Metal allocation limit during backward. The practical local training run uses `2560` context and `gradient_accumulation_steps=1`. CUDA Unsloth can use the larger effective batch through `--grad-accum 8`.

## 6. Train The Student

### Mac Path: MLX-Tune

On Apple Silicon, `train_unsloth.py` imports `mlx_tune` and keeps the training code shaped like Unsloth:

```bash
uv pip install mlx-tune

uv run python 1-distilling-a-0-8b-tool-calling-agent/train_unsloth.py \
  --train-path outputs/gpt_5_5_medium_sql_agent_train_sft_canonical_3072.jsonl \
  --output-dir outputs/qwen3_5_0_8b_mlx_tune_sql_agent_gpt_teacher_sft_2560_gradaccum1 \
  --max-seq-length 2560 \
  --grad-accum 1 \
  --learning-rate 1e-4
```

The script uses response-only training, so the loss is on the assistant JSON decision rather than on the whole prompt.

The same file can still run on CUDA later by forcing the CUDA backend:

```bash
uv pip install unsloth

uv run python 1-distilling-a-0-8b-tool-calling-agent/train_unsloth.py \
  --backend cuda \
  --model unsloth/Qwen3.5-0.8B \
  --train-path outputs/gpt_5_5_medium_sql_agent_train_sft_canonical_3072.jsonl \
  --output-dir outputs/qwen3_5_0_8b_unsloth_sql_agent_gpt_teacher_sft_3072 \
  --learning-rate 1e-4
```

## 7. Evaluate The Tuned Student

The evaluation harness should still go through BAML. Serve the tuned model through an OpenAI-compatible endpoint, then run the same eval script with `--model` and `--base-url`.

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
  --max-new-tokens 1024 \
  --output outputs/qwen3_5_0_8b_mlx_tune_sql_agent_gpt_teacher_sft_2560_gradaccum1_eval.json
```

On NVIDIA later, serve a CUDA Unsloth adapter with vLLM LoRA:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model unsloth/Qwen3.5-0.8B \
  --enable-lora \
  --lora-modules sql_agent=outputs/qwen3_5_0_8b_unsloth_sql_agent_gpt_teacher_sft_3072/adapter \
  --served-model-name qwen3_5_0_8b_sql_agent \
  --port 8094
```

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/eval_sql_agent.py \
  --model qwen3_5_0_8b_sql_agent \
  --base-url http://127.0.0.1:8094/v1 \
  --data-dir data/sql_agent_bird_critic \
  --max-turns 8 \
  --max-new-tokens 1024 \
  --output outputs/qwen3_5_0_8b_unsloth_sql_agent_gpt_teacher_sft_3072_eval.json
```

Measured result:

```text
No current BAML-harness tuned-student result is checked in yet.
```

Fill this in only after rerunning the tuned model through the same BAML structured-output harness.

## Result Summary

| Run | Success | Notes |
| --- | ---: | --- |
| Qwen3.5-0.8B base student | rerun | BAML harness |
| LFM2.5-8B-A1B baseline | rerun | BAML harness |
| Qwen3.5-35B-A3B teacher | rerun | BAML harness |
| GPT 5.5 medium teacher | rerun | BAML harness |
| Qwen3.5-0.8B tuned student | rerun | BAML harness |

## Next Fixes

- Run the current canonical 3072-token SFT file through the recommended Unsloth recipe.
- Add a format-only warmup dataset so the student reliably emits JSON actions before learning SQL.
- Consider training more LoRA layers or a stronger small student.
