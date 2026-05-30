# Distillation Blogs

This repo is a blog-series workspace for practical model distillation experiments.

The series goal is to make the tradeoffs visible: start with a small model, measure it in a real harness, generate teacher demonstrations, fine-tune the small model, then compare before and after. Later posts can reuse the same task family for softer distillation targets, on-policy correction, and reward-based training.

## Blogs

| Blog | Status | Focus |
| --- | --- | --- |
| [1. Distilling A Focused Function-Calling Model](1-distilling-a-0-8b-tool-calling-agent/) | active | Offline hard-token SFT from a stronger teacher into Qwen3.5 0.8B on a short-call NESTFUL slice. |

## Repo Layout

- `common/`: shared code used across blog posts.
- `data/`: prepared datasets and local dataset archives.
- `outputs/`: generated evals, teacher rows, adapters, and result summaries.
- `docs/plan.md`: current series plan and upcoming posts.
- `1-distilling-a-0-8b-tool-calling-agent/`: Blog 1 post, notebook, assets, scripts, and runbook.

## Project Conventions

Each blog folder owns its own runnable instructions. The root README stays as the high-level series map as more posts are added.

Generated files stay out of the blog folders unless they are assets used by the post. Large or repeated run artifacts belong in root `outputs/`; prepared datasets belong in root `data/`.

Secrets belong only in `.env`, which is ignored.

## Current Blog 1 Snapshot

Blog 1 uses `ibm-research/nestful`, filtered to examples with at most two expected function calls. The verified local path uses:

- student: `mlx-community/Qwen3.5-0.8B-MLX-bf16`
- main teacher for training data: `mlx-community/Qwen3.5-35B-A3B-8bit`
- training method: offline hard-token SFT with MLX-LM LoRA

Latest verified Blog 1 result: the student moved from `2/103` exact before SFT to `43/103` exact after SFT on the held-out eval split.
