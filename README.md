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

Current prepared split:

| Partition | Rows |
| --- | ---: |
| train | 879 |
| eval | 220 |

The split is percentage-based, not a fixed `500/100` sample: Notebook 01 filters to `Query` tasks in `netflix`, `movie_3`, `books`, and `chinook`, then keeps `20%` of each database for eval. Blog 1 treats Unsloth bf16 LoRA as the NVIDIA training path for a 16GB GPU, with MLX kept as the Apple experiment path.

Before long NVIDIA runs, use the Blog 1 README's GPU safety setup: cap the GPU at `150W`, enable the persistent power-limit service, and add swap on low-RAM machines.

## Folders

- `common/`: shared code for model generation, config, ChatGPT shim, and the SQL-agent harness.
- `1-distilling-a-0-8b-tool-calling-agent/`: Blog 1 scripts, notebooks, blog draft, and runbook.
- `data/`: prepared benchmark splits and SQLite templates.
- `outputs/`: eval reports, teacher traces, SFT rows, adapters, and summaries.

Blog-specific commands live in the blog folder README.
