# Blog 1: Distilling A 0.8B Tool-Calling Agent

This folder contains the teaching notebooks, assets, and runnable Python scripts for the first distillation post.

The notebooks are the tutorial. The scripts are the server path for running the experiment on rented NVIDIA GPUs.

## Scripts

Run these from the repo root.

```bash
uv run python 1-distilling-a-0-8b-tool-calling-agent/scripts/eval_student_hf.py
uv run python 1-distilling-a-0-8b-tool-calling-agent/scripts/eval_teacher.py
uv run python 1-distilling-a-0-8b-tool-calling-agent/scripts/collect_teacher_sft_rows.py
uv run python 1-distilling-a-0-8b-tool-calling-agent/scripts/train_student_trl.py
uv run python 1-distilling-a-0-8b-tool-calling-agent/scripts/eval_student_hf.py \
  --adapter outputs/qwen_qwen3_5_0_8b_tau3_retail_sft_trl_peft/qwen_qwen3_5_0_8b_tau3_retail_trl_lora_adapter
```

## Environment

For a remote GPU server, use a normal LiteLLM/OpenAI-compatible user simulator:

```bash
TAU_BENCH_USER_SIMULATOR_LLM=openai/gpt-5.4-mini
TAU_BENCH_USER_SIMULATOR_BACKEND=litellm
OPENAI_API_KEY=...
```

Start the local teacher endpoint with vLLM:

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

## Outputs

All generated artifacts still go to the repo-level `outputs/` folder:

- eval JSON files
- per-task local traces
- teacher train trajectories
- SFT JSONL rows
- TRL/PEFT adapter checkpoints
- training metadata

The blog folder should stay mostly notebooks, assets, and scripts. Data and model outputs should not be written here.
