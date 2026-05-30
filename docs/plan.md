# Distillation Blog Plan

The first blog now uses one deterministic nested function-calling dataset that can carry the whole distillation series.

## Dataset Choice

Dataset:

- `ibm-research/nestful`
- 1,861 total examples
- published as one split, so we create a deterministic local `80/20` train/eval split
- default split seed: `42`
- Blog 1 slice: reference call sequence length `<= 2`
- about 506 train rows and 103 eval rows for Blog 1

Task:

Given a user query and a catalog of available functions, the model must output a JSON sequence of function calls. Later calls can reference earlier outputs with labels such as `$var_1.result$`.

Why this is cleaner:

- It is normal function calling, not shopping/recommender behavior and not a UI action dataset.
- It is focused: the skill is nested/sequential API calling with variable wiring.
- It is deterministic and does not need an LLM user simulator.
- The `<= 2` slice is small enough to run often, but still leaves room for the 0.8B student to improve.
- It can support the whole series: hard-token SFT, logit distillation, on-policy correction, and later reward-based training using executable correctness.

## Blog 1

Question:

> Can a 0.8B model learn short nested function-call sequencing from a stronger teacher?

Main models:

- Student: `mlx-community/Qwen3.5-0.8B-MLX-bf16` for the verified local Blog 1 path.
- Main teacher: `mlx-community/Qwen3.5-35B-A3B-8bit` served through `mlx_lm.server`.
- Baseline teachers only: `gpt-5.5` with reasoning effort `medium` through the local ChatGPT subscription shim, and `LiquidAI/LFM2.5-8B-A1B-MLX-8bit` through MLX.

Blog 1 scripts:

- `prepare_nestful.py`: downloads and normalizes `ibm-research/nestful`.
- `eval_nestful.py`: runs exact nested-call sequence eval with HF, MLX, or OpenAI-compatible inference.
- `generate_teacher_sft_rows.py`: runs the teacher on the train split and keeps exact successful call sequences.
- `train_mlx.py`: MLX-LM LoRA path used in the verified Blog 1 run.
- `train_trl.py` and `train_unsloth.py`: kept as later comparison paths, not part of the current verified runbook.

Metrics:

1. Exact sequence accuracy: function names, labels, and arguments all match.
2. Name-sequence accuracy: the right functions are selected in the right order.
3. Parse rate: the model emitted valid JSON in the expected structure.

Current verified results on the `<= 2` eval split with parser v2:

- Qwen3.5-0.8B MLX student before SFT: `2/103` exact, `7/103` name-sequence, `83/103` parseable.
- Qwen3.5-35B-A3B 8-bit MLX-server teacher: `37/103` exact, `39/103` name-sequence, `98/103` parseable.
- GPT 5.5 medium teacher baseline: `69/103` exact, `78/103` name-sequence, `103/103` parseable.
- LFM2.5-8B-A1B MLX 8-bit baseline: `0/103` exact, `0/103` name-sequence, `0/103` parseable.
- Qwen3.5-0.8B MLX student after Qwen-teacher SFT: `43/103` exact, `53/103` name-sequence, `99/103` parseable.

Teacher train generation with the Qwen teacher kept `156/506` exact rows for SFT.

## Later Blogs

2. Soft-label/logit distillation on the same NESTFUL prompts and completions.
3. On-policy distillation: let the student produce call sequences, then use teacher/verifier feedback to create corrections.
4. RL/GRPO-style training: use executable answer correctness from NESTFUL functions as reward.
