---
name: slm-trainer
description: Plan, run, debug, and report small-language-model distillation and LoRA fine-tuning experiments. Use when working on SLM/student-teacher training, agent harness distillation, Unsloth/TRL/MLX training, teacher trace generation, eval parity, GPU-server training, model artifact sync, or diagnosing why a fine-tuned small model did not improve.
---

# SLM Trainer

Use this skill to keep small-model distillation experiments honest, reproducible, and useful. The default stance is: first prove the harness and data are comparable, then train, then trust only full task evals.

## Workflow

1. Pin the experiment contract.
   - Name the task, dataset split, harness version, parser, tool schema, model input format, and scoring path.
   - Confirm train/eval split sizes from files, not memory.
   - Refuse to compare runs that used different harness contracts unless the difference is the experiment.

2. Freeze teacher data before training.
   - Generate teacher traces through the same harness the student will use.
   - Keep only trajectories that passed the deterministic task scorer unless the experiment explicitly studies failed/corrected traces.
   - Write immutable timestamped artifacts plus a `current` alias.
   - Record attempted tasks, successful tasks, trace rows, model, backend, decoding config, prompt/harness version, and row/token retention.

3. Check train/eval parity.
   - The student training target must be exactly what the eval parser expects.
   - For structured harnesses, train on the canonical action string consumed by the harness, not on a teacher blob with extra formatting.
   - If BAML or another parser formats the model output, the training examples should teach that same formatted action contract.

4. Measure token lengths before choosing context.
   - Tokenize the full prompt plus target with the actual student tokenizer.
   - Report source rows, kept rows, dropped rows, train/validation rows, p50/p90/p95/max tokens.
   - Pick the smallest context that preserves the rows needed for the experiment. Do not silently drop many rows.

5. Train with a boring first recipe.
   - Prefer full-precision or bf16 LoRA when memory allows; use QLoRA only when required by memory.
   - Start with batch size 1, gradient accumulation 4-8, learning rate around `5e-5` for LoRA SFT, and rank/alpha 16 or 32.
   - Use gradient checkpointing when it is required for memory. Verify FlashAttention is actually active on CUDA.
   - Save `training_config.json` with every effective argument and data statistic.
   - Keep enough checkpoints to evaluate likely early-stopping candidates, or explicitly save/load the best checkpoint. Do not rely on validation loss if `save_total_limit` already deleted the best checkpoint.

6. Evaluate like production.
   - Run the full eval split, not smoke tests, for reported numbers.
   - Use the same prompt, parser, tool execution, deterministic scorer, decoding config, and max-turn policy as the baseline.
   - Report success, submitted count, parse failures, repeated-action failures, runtime errors, SQL/tool errors, and average turns.
   - Sync the final eval JSON/log before editing or resetting a remote pod.

7. Interpret with discipline.
   - Validation loss is useful but not decisive for agents. Full task success wins.
   - If submission rate improves but task success does not, the model learned harness control but not enough task reasoning.
   - If parse failures dominate, fix data/harness parity before changing model size.
   - If SQL/tool errors dominate, inspect failed traces and teacher quality before adding more epochs.
   - If more data lowers validation loss but lowers task success, do failure analysis before collecting more data.

## Remote GPU Rules

- Keep generated data and adapters under persistent storage such as `/workspace` when possible.
- If the container root disk is full, set `TMPDIR` to a writable high-capacity path such as `/dev/shm/...` before running eval/training.
- Do not put executable compiler caches on `noexec` mounts. `/dev/shm` may be good for temporary data but bad for Triton `.so` loading if mounted `noexec`.
- If a provider workspace hits quota, do not assume `df -h` is enough; inspect per-directory usage and provider quota behavior before rerunning.
- When root is full and `/workspace` is at file quota, reusing existing root compile caches can be safer than launching new cache variants. If a new cache must be created, ask before deleting old caches.
- On unstable personal GPU machines, cap power with `nvidia-smi -pl ...` for stability. This limits power/heat, not VRAM.
- Do not wait blindly after GPU faults. If SSH or `nvidia-smi` stays broken after a reboot-worthy fault, ask the user to reboot or power-cycle.
- Before resizing or editing a RunPod pod, sync eval JSON, logs, adapters, and training configs. Editing a running pod can reset it and lose non-volume data.

## Unsloth And Model Notes

- CUDA Unsloth is the preferred NVIDIA path when available; verify the installed stack with a tiny real train run before long jobs.
- MLX/MLX-Tune can be useful on Apple Silicon, but confirm gradient accumulation is actually forwarded and expect much slower backward passes than CUDA.
- Use the same code path where possible, but do not hide backend-specific differences. Record the backend in every output config.
- For Qwen-style dense models, common LoRA targets are `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- For LFM2.5 MoE models in Unsloth-style fine-tuning, use LFM-compatible targets such as `in_proj`, `out_proj`, `q_proj`, `k_proj`, `v_proj`, `w1`, `w2`, `w3`.
- For MoE models, verify the experts backend. `grouped_mm` can be fastest but hardware-dependent; `batched_mm` can duplicate selected expert weights and explode memory; `eager` is slower but often the safest compatibility path.

## Reporting Template

Always include:

```text
Teacher data:
  model/backend:
  attempted tasks:
  successful tasks:
  SFT rows:
  frozen files:

Training:
  student model:
  rows kept/dropped:
  max sequence length:
  batch/grad accumulation:
  learning rate:
  LoRA rank/alpha:
  steps/epochs:
  runtime:
  validation loss:

Eval:
  split:
  success:
  submitted:
  parse failures:
  repeated-action failures:
  runtime errors:
  artifact path:

Interpretation:
  what improved:
  what did not improve:
  next investigation:
```

## Guardrails

- Never report partial eval progress as a final score.
- Never mix old-data and new-data results without labeling them.
- Never train on data generated by a different parser/harness unless that mismatch is intentional and stated.
- Never regenerate teacher data when frozen successful traces already answer the experiment.
- Never use hidden regex or phrase hacks to make a benchmark pass.
- Never hide negative results; they are often the useful part of the experiment.
