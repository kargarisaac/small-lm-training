# Distillation Blogs

Notebook-first tutorials plus simple runnable experiment runners for distilling a small retail tool-calling agent.

The notebooks stay in `1-distilling-a-0-8b-tool-calling-agent/` for teaching. The runners in `1-distilling-a-0-8b-tool-calling-agent/runners/` are the clean server workflow for blog one: run baseline evals, collect teacher trajectories, train with TRL/PEFT or MLX, then evaluate the trained adapter.

Generated data lives in root `outputs/`. Benchmark source/cache data lives in root `data/`.

## Server Setup

On a rented NVIDIA box:

```bash
git clone <your-repo-url>
cd distillation-blogs
uv sync
uv pip install vllm
```

Create `.env`:

```bash
TAU_BENCH_USER_SIMULATOR_LLM=openai/gpt-5.4-mini
TAU_BENCH_USER_SIMULATOR_BACKEND=litellm
OPENAI_API_KEY=...
```

`TAU_BENCH_USER_SIMULATOR_BACKEND=litellm` is the remote-server path. The old local ChatGPT subscription shim remains the default if this env var is not set, so the notebooks still work locally as before.

## Teacher Server

Start a vLLM teacher in another terminal. For NVIDIA, use the official HF model, not the MLX conversion:

```bash
uv run vllm serve Qwen/Qwen3.5-35B-A3B \
  --host 127.0.0.1 \
  --port 8092 \
  --served-model-name Qwen/Qwen3.5-35B-A3B \
  --max-model-len 81920 \
  --dtype bfloat16 \
  --trust-remote-code \
  --generation-config vllm
```

If memory is tight, switch to an HF quantized teacher and pass that same name to the runners with `--model`.

## Run Order

Baseline student on held-out test tasks with HF/PyTorch:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/runners/eval_student_hf.py
```

Baseline student on held-out test tasks with MLX-LM:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/runners/eval_student_mlx.py
```

Teacher on held-out test tasks:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/runners/eval_teacher.py
```

Teacher on train tasks, then extract successful SFT rows:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/runners/collect_teacher_sft_rows.py
```

Train the small student with TRL/PEFT:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/runners/train_student_trl.py
```

Train the small student with MLX-LM:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/runners/train_student_mlx.py
```

Evaluate the trained adapter:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/runners/eval_student_hf.py \
  --adapter outputs/qwen_qwen3_5_0_8b_tau3_retail_sft_trl_peft/qwen_qwen3_5_0_8b_tau3_retail_trl_lora_adapter
```

Evaluate an MLX adapter:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/runners/eval_student_mlx.py \
  --adapter outputs/mlx_community_qwen3_5_0_8b_mlx_bf16_tau3_retail_sft_mlx_lm/mlx_community_qwen3_5_0_8b_mlx_bf16_tau3_retail_mlx_lora_adapter
```

Summarize trace stats:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/runners/trace_stats.py
```

## Useful Options

Run a small smoke slice:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/runners/eval_teacher.py --limit 5
uv run python 1-distilling-a-0-8b-tool-calling-agent/runners/collect_teacher_sft_rows.py --limit 5
```

Resume TRL training from the latest saved checkpoint:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/runners/train_student_trl.py --resume latest
```

Change batch size without editing code:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/runners/train_student_trl.py --batch-size 8
```

Enable MLflow logging during eval:

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/runners/eval_student_hf.py --mlflow
uv run python 1-distilling-a-0-8b-tool-calling-agent/runners/eval_teacher.py --mlflow
```

## Script Defaults

The server runners are intentionally plain:

- student: `Qwen/Qwen3.5-0.8B`
- teacher: `Qwen/Qwen3.5-35B-A3B`
- prompt tokenizer: `Qwen/Qwen3.5-0.8B`
- benchmark: tau3-bench retail via the pinned `tau2-bench` revision in `common/config.py`
- eval split: all `test` tasks
- teacher data split: all `train` tasks
- max steps per task: `100`
- max new tokens per model action: `2048`
- TRL max sequence length: `16500`
- TRL batch size: `4`
- LoRA: final transformer layer only, rank `8`, alpha `20`, target attention and MLP projection modules

The notebooks still contain the teaching walkthroughs, diagrams, and local MLX experiments. The runners are the production-ish path for running the experiment on a GPU server.

## Data Transfer

`outputs/` and `.env*` files are ignored. Do not push secrets or local traces to GitHub.

For server training without rerunning teacher trajectory collection, copy only the SFT rows JSONL:

```bash
rsync -av outputs/*_tau3_bench_retail_train_successful_sft_chat_rows_*.jsonl \
  user@server:/path/to/distillation-blogs/outputs/
```

The pinned tau2-bench checkout under `data/external/` is also ignored and can be recreated by the runners.
