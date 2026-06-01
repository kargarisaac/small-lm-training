# Distillation Blogs

This repo is a hands-on series about specializing small models for tool-use harnesses.

The point is not only "small model imitates big model." For agentic systems, the model lives behind an interface: prompt builder, parser, tools, environment observations, and deterministic scoring or reward. The series studies how to distill behavior inside that harness.

## Posts

| Post | Status | Topic |
| --- | --- | --- |
| [1. Distilling A 0.8B SQL Tool-Use Agent](1-distilling-a-0-8b-tool-calling-agent/) | active | Offline hard-token distillation for a multi-turn SQLite agent harness. |

## Current Blog 1 Setup

Blog 1 now uses `birdsql/six-gym-sqlite`, a BIRD-Critic/SIX-GYM SQLite dataset with real databases, SQL test cases, and deterministic scoring.

The prepared SQLite templates are about `554 MB`; two files exceed GitHub's normal file-size limit, so the repo keeps the small split JSONL files and regenerates/downloads database templates from Blog 1 Notebook 01.

The harness loop is:

```text
question -> model JSON action -> SQLite tool result -> next model action -> submit SQL -> tests
```

Previous full eval results on the old 100-row held-out split:

| Run | Success |
| --- | ---: |
| Qwen3.5-0.8B MLX base student | 4/100 |
| LFM2.5-8B-A1B MLX 8-bit baseline | 0/100 |
| Qwen3.5-35B-A3B 8-bit MLX-server teacher | 33/100 |
| GPT 5.5 medium teacher | 51/100 |
| Qwen3.5-0.8B after first partial GPT-SFT run | 1/100 |

The first partial SFT attempt did **not** improve the student. The next training recipe uses the completed GPT teacher set: 242 successful trajectories, 767 SFT rows, canonical one-action labels, and a 3072-token default that keeps 737 rows. Blog 1 now treats vLLM as the NVIDIA serving path and Unsloth bf16 LoRA as the recommended NVIDIA training path for a 16GB GPU, with MLX and TRL kept as Apple/reference paths.

## Folders

- `common/`: shared code for model generation, config, ChatGPT shim, and the SQL-agent harness.
- `1-distilling-a-0-8b-tool-calling-agent/`: Blog 1 scripts, notebooks, blog draft, and runbook.
- `data/`: prepared benchmark splits and SQLite templates.
- `outputs/`: eval reports, teacher traces, SFT rows, adapters, and summaries.

Blog-specific commands live in the blog folder README.
