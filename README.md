# Distillation Blogs

Notebook-first tutorials for distilling a small Qwen tool-calling model from a stronger same-family teacher.

## Teacher Server

Blog 1 uses MLX-LM to serve the teacher model locally:

```bash
uv run python scripts/serve_teacher_mlx.py
```

The default teacher artifact is:

```text
mlx-community/Qwen3.5-35B-A3B-4bit
```

The first server run downloads the model into the Hugging Face cache. The notebook calls `http://127.0.0.1:8080/v1/completions` with an already-rendered Qwen prompt.
