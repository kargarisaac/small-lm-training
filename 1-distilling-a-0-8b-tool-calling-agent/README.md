# Blog 1: Distilling A Focused Function-Calling Model

This folder contains the Blog 1 post, teaching notebook, assets, and runnable scripts.

The experiment asks:

> Can a 0.8B model learn short nested function-call sequencing from a stronger teacher?

The verified local path uses:

- **student:** `mlx-community/Qwen3.5-0.8B-MLX-bf16`
- **main teacher for training data:** `mlx-community/Qwen3.5-35B-A3B-8bit`, served through `mlx_lm.server`
- **teacher baselines only:** `gpt-5.5` medium through the local ChatGPT subscription shim, and `LiquidAI/LFM2.5-8B-A1B-MLX-8bit`
- **training method:** offline hard-token SFT with MLX-LM LoRA

## Files

- Blog markdown: [blog.md](blog.md)
- Teaching notebook: [notebooks/01_explore_nestful_short.ipynb](notebooks/01_explore_nestful_short.ipynb)
- Dataset preparation: [prepare_nestful.py](prepare_nestful.py)
- Evaluation: [eval_nestful.py](eval_nestful.py)
- Teacher row generation: [generate_teacher_sft_rows.py](generate_teacher_sft_rows.py)
- MLX training: [train_mlx.py](train_mlx.py)
- ChatGPT subscription shim: [serve_chatgpt_shim.py](serve_chatgpt_shim.py)

## Backend Meaning

- `mlx`: the script loads a local MLX-LM model itself.
- `openai`: the script calls an OpenAI-compatible HTTP server at `/v1/chat/completions`. In this runbook that means either `mlx_lm.server` for the local Qwen teacher, or the local ChatGPT subscription shim for GPT 5.5.
- `hf`: supported by the scripts, but not part of the verified local Blog 1 run.

Run all commands from the repo root.

## 0. Setup

```bash
uv sync
```

## 1. Prepare The Dataset

Blog 1 uses `ibm-research/nestful`, filtered to reference solutions with at most two function calls.

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/prepare_nestful.py --max-calls 2
```

Output:

```text
data/nestful_calls_le_2/train.jsonl
data/nestful_calls_le_2/eval.jsonl
data/nestful_calls_le_2/stats.json
```

Expected split:

- train rows: `506`
- eval rows: `103`
- split seed: `42`
- eval fraction: `0.2`

## 2. Run The Student Baseline

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/eval_nestful.py \
  --backend mlx \
  --model mlx-community/Qwen3.5-0.8B-MLX-bf16 \
  --data-dir data/nestful_calls_le_2 \
  --output outputs/qwen3_5_0_8b_mlx_nestful_calls_le_2_eval_parser_v2.json
```

This evaluates the untuned small model on the held-out eval split.

## 3. Run Teacher Baselines

Only the Qwen teacher is used for training data in this post. GPT 5.5 and LFM are eval baselines for comparison.

### 3.1 Qwen Teacher Through MLX-LM Server

Start the Qwen teacher server in a separate terminal:

```bash
uv run mlx_lm.server \
  --model mlx-community/Qwen3.5-35B-A3B-8bit \
  --host 127.0.0.1 \
  --port 8092 \
  --max-tokens 4096 \
  --temp 0 \
  --top-p 1 \
  --top-k 0 \
  --chat-template-args '{"enable_thinking": false}'
```

Then evaluate the teacher:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/eval_nestful.py \
  --backend openai \
  --model mlx-community/Qwen3.5-35B-A3B-8bit \
  --base-url http://127.0.0.1:8092/v1 \
  --data-dir data/nestful_calls_le_2 \
  --output outputs/qwen3_5_35b_a3b_8bit_mlx_server_nestful_calls_le_2_eval_parser_v2.json
```

### 3.2 GPT 5.5 Medium Baseline

Start the local ChatGPT subscription shim in a separate terminal:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/serve_chatgpt_shim.py \
  --model gpt-5.5 \
  --reasoning-effort medium \
  --port 8080
```

Then evaluate GPT 5.5 medium:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/eval_nestful.py \
  --backend openai \
  --model gpt-5.5 \
  --base-url http://127.0.0.1:8080/v1 \
  --reasoning-effort medium \
  --data-dir data/nestful_calls_le_2 \
  --output outputs/gpt_5_5_medium_nestful_calls_le_2_eval_parser_v2.json
```

### 3.3 LFM2.5 8B A1B Baseline

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/eval_nestful.py \
  --backend mlx \
  --model LiquidAI/LFM2.5-8B-A1B-MLX-8bit \
  --data-dir data/nestful_calls_le_2 \
  --output outputs/lfm2_5_8b_a1b_mlx_nestful_calls_le_2_eval_parser_v2.json
```

## 4. Generate Qwen Teacher SFT Rows

Keep the Qwen MLX-LM server from Step 3.1 running.

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/generate_teacher_sft_rows.py \
  --backend openai \
  --model mlx-community/Qwen3.5-35B-A3B-8bit \
  --base-url http://127.0.0.1:8092/v1 \
  --data-dir data/nestful_calls_le_2 \
  --output outputs/qwen3_5_35b_a3b_8bit_mlx_server_nestful_calls_le_2_train_teacher_sft_rows_parser_v2.jsonl
```

The JSONL keeps only train rows where the teacher exactly matched the reference call sequence.

Output:

```text
outputs/qwen3_5_35b_a3b_8bit_mlx_server_nestful_calls_le_2_train_teacher_sft_rows_parser_v2.jsonl
outputs/qwen3_5_35b_a3b_8bit_mlx_server_nestful_calls_le_2_train_teacher_sft_rows_parser_v2.report.json
```

## 5. Train The Student

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/train_mlx.py \
  --model mlx-community/Qwen3.5-0.8B-MLX-bf16 \
  --train-path outputs/qwen3_5_35b_a3b_8bit_mlx_server_nestful_calls_le_2_train_teacher_sft_rows_parser_v2.jsonl \
  --output-dir outputs/qwen3_5_0_8b_mlx_nestful_calls_le_2_qwen_teacher_sft_parser_v2_b1 \
  --batch-size 1 \
  --grad-accum 1 \
  --learning-rate 1e-5 \
  --iters 100 \
  --max-seq-length 3072
```

Final adapter:

```text
outputs/qwen3_5_0_8b_mlx_nestful_calls_le_2_qwen_teacher_sft_parser_v2_b1/adapter/adapters.safetensors
```

## 6. Evaluate The Trained Student

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/eval_nestful.py \
  --backend mlx \
  --model mlx-community/Qwen3.5-0.8B-MLX-bf16 \
  --adapter outputs/qwen3_5_0_8b_mlx_nestful_calls_le_2_qwen_teacher_sft_parser_v2_b1/adapter \
  --data-dir data/nestful_calls_le_2 \
  --output outputs/qwen3_5_0_8b_mlx_nestful_calls_le_2_qwen_teacher_sft_parser_v2_b1_eval.json
```

## Latest Verified Numbers

Verified on May 30, 2026 with `data/nestful_calls_le_2` and parser v2.

| Run | Exact | Name Sequence | Parse Rate |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B MLX baseline | 2/103 | 7/103 | 83/103 |
| Qwen3.5-35B-A3B 8-bit teacher, MLX server | 37/103 | 39/103 | 98/103 |
| GPT 5.5 medium teacher | 69/103 | 78/103 | 103/103 |
| LFM2.5-8B-A1B MLX 8-bit baseline | 0/103 | 0/103 | 0/103 |
| Qwen3.5-0.8B MLX after Qwen-teacher SFT | 43/103 | 53/103 | 99/103 |

The compact result summary is:

```text
outputs/blog1_results_summary.json
```
