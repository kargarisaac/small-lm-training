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

On Apple Silicon, start the MLX server before using the BAML VSCode playground or `baml-cli test`:

```bash
uv run mlx_lm.server \
  --model mlx-community/Qwen3.5-0.8B-MLX-bf16 \
  --host 127.0.0.1 \
  --port 8091 \
  --chat-template-args '{"enable_thinking": false}'
```

On NVIDIA/CUDA, serve the Hugging Face model through vLLM instead. Install vLLM once as a user-level `uv` tool so all projects can reuse the same isolated vLLM command without sharing vLLM's stricter CUDA dependency set with each notebook or training environment:

```bash
uv tool install --python 3.12 "vllm>=0.22.0"

VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen3.5-0.8B \
  --host 127.0.0.1 \
  --port 8091 \
  --served-model-name Qwen/Qwen3.5-0.8B \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

`VLLM_USE_FLASHINFER_SAMPLER=0` avoids requiring a system `nvcc` install on machines where the vLLM wheel and NVIDIA driver are present but `/usr/local/cuda` is not.

`Qwen/Qwen3.5-0.8B` uses Qwen3.5's GDN/linear-attention path. If vLLM reports a GDN or Triton CUDA launch failure on an Ada GPU such as an RTX 4080, stop retrying that same process; reset or reboot the GPU before starting another CUDA server, then use a newer vLLM build or a non-vLLM Transformers server for this architecture.

Then run:

```bash
uv run baml-cli test -i "SqlAgentNextAction::"
```

Configured BAML clients:

| BAML client | Model | Server to start |
| --- | --- | --- |
| `MlxCommunityQwen35_08bMlxBf16` | `mlx-community/Qwen3.5-0.8B-MLX-bf16` | `mlx_lm.server` on `8091` |
| dynamic notebook/eval client | `Qwen/Qwen3.5-0.8B` | vLLM OpenAI server on `8091` |
| `MlxCommunityQwen35_35bA3b8bit` | `mlx-community/Qwen3.5-35B-A3B-8bit` | `mlx_lm.server` on `8092` |
| `LiquidAiLfm25_8bA1bMlx8bit` | `LiquidAI/LFM2.5-8B-A1B-MLX-8bit` | `mlx_lm.server` on `8093` |
| `Gpt55Medium` | `gpt-5.5` | local ChatGPT shim on `8080` |
| `Qwen35_08bSqlAgentVllm` | `qwen3_5_0_8b_sql_agent` | vLLM OpenAI server on `8094` |
| `MlxCommunityQwen35_08bMlxBf16Lora` | `mlx-community/Qwen3.5-0.8B-MLX-bf16` with adapter | `mlx_lm.server --adapter-path ...` on `8095` |

The Python eval scripts and Notebook 01 choose a model dynamically with `--model`/`model_name` and `--base-url`/`base_url`, so they can use either MLX or vLLM. The BAML VSCode extension uses the static client selected in `baml_src/sql_agent.baml`, so that server must be running first.

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
| Mac | `mlx-lm` or `mlx-tune` | local LoRA experiments |
| NVIDIA | Unsloth or TRL/PEFT | main training runs |

For Blog 1 hard-token SFT, we train on canonical BAML decisions:

```json
{"draft": "Need schema first.", "output": {"action": "inspect_schema"}}
```

For a later soft-label/logit post, BAML is still useful for deciding the canonical target string, but BAML itself does not give token probabilities. We would score the canonical BAML target under the teacher model in teacher-forcing/scoring mode.

Teacher forcing here means: instead of asking the teacher to generate the answer freely, we feed the prompt plus the known target tokens to the model and ask, token by token, what probability it assigned to each target token. That needs a serving/scoring path that exposes logprobs, such as vLLM or a direct HF forward pass.

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

## 2. Evaluate The Base Student

### NVIDIA Path: vLLM

Serve the Hugging Face model with the OpenAI-compatible vLLM server:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen3.5-0.8B \
  --host 127.0.0.1 \
  --port 8091 \
  --served-model-name Qwen/Qwen3.5-0.8B \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

The important serving/eval numbers are different budgets:

```text
--max-model-len 32768
  vLLM's total context window per request: prompt tokens plus generated tokens.
  SQL-agent prompts can become large after schema observations, so use a larger
  server window when allowing 4096 generated tokens.

--max-new-tokens 4096
  Maximum generated tokens for one assistant action. The model usually emits
  far less because each turn should be one JSON decision, but this leaves enough
  room for long final SQL submissions.

--max-turns 8
  Maximum SQL-agent loop steps for one benchmark row. It is not a token setting.
```

For every model call, `prompt_tokens + max_new_tokens` must fit inside `max_model_len`. Do not pair `--max-model-len 4096` with `--max-new-tokens 4096`; that leaves no space for the prompt and will fail even if the server is reachable.

Run the BAML harness eval:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/eval_sql_agent.py \
  --model Qwen/Qwen3.5-0.8B \
  --base-url http://127.0.0.1:8091/v1 \
  --data-dir data/sql_agent_bird_critic \
  --max-turns 8 \
  --max-new-tokens 4096 \
  --output outputs/qwen3_5_0_8b_vllm_sql_agent_eval.json
```

Notebook 01 uses the same local server in its optional live one-task run. The notebook writes `train.jsonl`, `eval.jsonl`, `stats.json`, and downloads the SQLite templates before the live harness call needs them.

### Apple Path: MLX

```bash
uv run mlx_lm.server \
  --model mlx-community/Qwen3.5-0.8B-MLX-bf16 \
  --host 127.0.0.1 \
  --port 8091 \
  --chat-template-args '{"enable_thinking": false}'
```

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

Previous completed GPT teacher generation, before the percentage-based domain split:

```text
completed train tasks: 500/500
successful trajectories: 242/500
BAML-canonical SFT trace rows: 767
canonical rows <= 3072 tokens: 737
canonical rows <= 4096 tokens: 754
```

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

At `3072`, the canonical GPT-teacher SFT file keeps `737/767` rows. If a 16GB GPU still OOMs, drop to `2560`, which keeps `679/767` rows.

## 6. Train The Student

### Recommended NVIDIA Path: Unsloth

Install Unsloth on the CUDA/Linux server, then run:

```bash
uv pip install unsloth

uv run python 1-distilling-a-0-8b-tool-calling-agent/train_unsloth.py \
  --model unsloth/Qwen3.5-0.8B \
  --train-path outputs/gpt_5_5_medium_sql_agent_train_sft_canonical_3072.jsonl \
  --output-dir outputs/qwen3_5_0_8b_unsloth_sql_agent_gpt_teacher_sft_3072
```

This is the recommended path for a rented RTX 4080 16GB. It uses bf16 LoRA by default. `--load-in-4bit` exists as a fallback if bf16 LoRA does not fit, but it is not the first choice for Qwen3.5.

### Apple Path: MLX-LM

Direct MLX inference uses `enable_thinking=false`. MLX-LM LoRA does not currently expose the same chat-template option in its trainer, so use this path for Apple-side experiments and prefer Unsloth/TRL when you need exact no-thinking SFT parity.

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/train_mlx.py \
  --model mlx-community/Qwen3.5-0.8B-MLX-bf16 \
  --train-path outputs/gpt_5_5_medium_sql_agent_train_sft_canonical_3072.jsonl \
  --output-dir outputs/qwen3_5_0_8b_mlx_sql_agent_gpt_teacher_sft_3072
```

### Reference NVIDIA Path: TRL/PEFT

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/train_trl.py \
  --model Qwen/Qwen3.5-0.8B \
  --train-path outputs/gpt_5_5_medium_sql_agent_train_sft_canonical_3072.jsonl \
  --output-dir outputs/qwen3_5_0_8b_trl_sql_agent_gpt_teacher_sft_3072
```

TRL is kept because later soft-label/logit distillation will likely need a more standard PyTorch training loop. For Blog 1 hard-token SFT on a 16GB NVIDIA GPU, Unsloth is the preferred path.

## 7. Evaluate The Tuned Student

The evaluation harness should still go through BAML. Serve the tuned model through an OpenAI-compatible endpoint, then run the same eval script with `--model` and `--base-url`.

For an Unsloth/TRL adapter on the GPU server, one option is vLLM with LoRA enabled:

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

For an MLX adapter on the Mac:

```bash
uv run mlx_lm.server \
  --model mlx-community/Qwen3.5-0.8B-MLX-bf16 \
  --adapter-path outputs/qwen3_5_0_8b_mlx_sql_agent_gpt_teacher_sft_3072/adapter \
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
  --output outputs/qwen3_5_0_8b_mlx_sql_agent_gpt_teacher_sft_3072_eval.json
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

- Run the 737-row canonical 3072-token SFT file through the recommended Unsloth recipe.
- Add a format-only warmup dataset so the student reliably emits JSON actions before learning SQL.
- Compare Unsloth against the TRL reference path on the same config.
- Consider training more LoRA layers or a stronger small student.
