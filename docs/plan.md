# Distillation Blog Plan

## Direction

The series is centered on a deterministic multi-turn SQL-agent harness. Earlier one-turn function-plan datasets were dropped from the main story because the model did not receive observations and act again.

Blog 1 now uses:

```text
birdsql/six-gym-sqlite
```

The harness is:

```text
user SQL issue
-> BAML model call returns {draft, output}
-> inspect_schema / run_sql_query / submit_sql
-> SQLite result or error
-> next BAML model call
-> deterministic test-case score
```

No LLM user simulator. No LLM judge. The environment is SQLite plus test cases.

## Blog 1 Question

> Can a 0.8B model learn to behave as a small SQL tool-use agent from stronger teacher trajectories?

## Current Models

- Student: `mlx-community/Qwen3.5-0.8B-MLX-bf16`
- Local teacher baseline: `mlx-community/Qwen3.5-35B-A3B-8bit` through `mlx_lm.server`
- GPT teacher baseline and first SFT teacher: `gpt-5.5` with reasoning effort `medium`
- Additional baseline: `LiquidAI/LFM2.5-8B-A1B-MLX-8bit`
- Qwen chat-template thinking is disabled for student inference, Qwen teacher requests, and SFT tokenization.
- OpenAI-compatible inference now goes through BAML with a short `draft` field and one executable `output` action.

## Current Scripts

- `notebooks/01_explore_sql_agent_benchmark.ipynb`: explores the benchmark, filters to a focused `Query` database cluster, creates a percentage-based stratified train/eval split by database, and writes prepared data.
- `eval_sql_agent.py`: runs a served model through the BAML SQL-agent harness.
- `generate_sql_teacher_sft_rows.py`: runs the teacher on train tasks and writes BAML-canonical SFT trace rows from successful trajectories.
- `notebooks/02_explore_teacher_sft_data.ipynb`: explores BAML-canonical SFT trace rows, canonicalizes one-action targets, filters by token length, and writes the final SFT file.
- `train_unsloth.py`: one Unsloth-style LoRA path. It uses MLX-Tune on Apple Silicon and core Unsloth on CUDA, consuming the final SFT file from Notebook 02.

## Previous Measured Results

These are from the earlier fixed-count split. After the percentage-based domain split is rerun, replace this table with the new baseline numbers.

| Run | Success | Submitted | Parse Failures | Max-Turn Failures |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B base student | 4/100 | 19/100 | 76 | 5 |
| LFM2.5-8B-A1B baseline | 0/100 | 5/100 | 93 | 2 |
| Qwen3.5-35B-A3B 8-bit teacher | 33/100 | 81/100 | 5 | 14 |
| GPT 5.5 medium teacher | 51/100 | 99/100 | 0 | 1 |
| Qwen3.5-0.8B after first partial GPT SFT | 1/100 | 7/100 | 92 | 1 |

## Previous Teacher SFT Data

This was generated from the earlier fixed-count split. After rerunning Notebook 01 with the percentage-based domain split, regenerate teacher rows from the new train file.

```text
train tasks completed: 500/500
successful trajectories: 242/500
BAML-canonical SFT trace rows: 767
canonical rows <= 2560 tokens: 679
canonical rows <= 3072 tokens: 737
canonical rows <= 4096 tokens: 754
```

Recommended CUDA training config:

```text
Unsloth bf16 LoRA on NVIDIA first
max seq length: 3072
batch size: 1
gradient accumulation: 8
learning rate: 1e-5
lora rank: 16
lora alpha: 16
```

Earlier partial MLX SFT result:

```text
base student: 4/100
partial tuned student: 1/100
```

That negative result showed the pipeline works but the first data/training recipe was not good enough.

Current Mac training path uses the same `train_unsloth.py` entrypoint through MLX-Tune. In this environment the upstream MLX-Tune native trainer accepted `gradient_accumulation_steps` but did not forward it into MLX-LM `TrainingArgs`; we patched the installed package and `train_unsloth.py` now checks for that patch before Mac training. MLX-LM `iters` are microsteps, not optimizer updates, so gradient accumulation changes update frequency but does not reduce the number of examples processed in one pass. Treat Mac sequence length as an explicit speed/data-coverage tradeoff:

```text
2048 tokens: 471/721 rows fit
2560 tokens: 615/721 rows fit
3072 tokens: 721/721 rows fit but currently hits the Metal allocation limit
```

## Next Technical Fixes

1. Make GPT teacher generation robust enough to finish the full prepared train split, or switch train generation to a local teacher that cannot hang on subscription streaming.
2. Clean successful teacher traces so each SFT target is the canonical next action expected by the harness, not a non-canonical multi-action teacher blob.
3. For Blog 1, prefer shorter verified successful trajectories when multiple teacher traces solve the same task. The point is not to truncate context blindly; it is to train the student on clean, replayable paths with fewer wasted tool calls and fewer tokens.
4. Add a format-only warmup corpus so the student reliably emits exactly one JSON action per turn.
5. Train on many more successful trajectories.
6. Keep the single `train_unsloth.py` training entrypoint portable: MLX-Tune on Apple Silicon now, core Unsloth on CUDA later.
7. Consider more LoRA layers or a stronger student model.

## Future Posts

1. **Blog 1:** hard-token offline distillation in the SQL-agent harness. Start with verifier-filtered successful teacher traces, and when possible select shorter successful traces so the small model learns the clean execution path rather than the teacher's wandering.
2. **Blog 2:** trajectory pruning and compaction for small agent models. Take successful traces, remove or summarize redundant observations, preserve the state needed for the next action, replay or verify the compacted trajectory, then train on the compact state/action pairs.
3. **Blog 3:** soft-label/logit distillation on the same compacted teacher actions.
4. **Blog 4:** on-policy distillation: let the student act, collect its bad states, and ask the teacher/verifier for corrections.
5. **Blog 5:** RL/GRPO-style training using SQLite test-case success as reward.
