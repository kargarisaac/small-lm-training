# Distillation Blogs

Notebook-first tutorials for distilling a small Qwen tool-calling model from a stronger same-family teacher.

Shared notebook/runtime code lives in `common/`. Blog folders should contain notebooks and local assets only; benchmark data lives under root `data/`, and generated artifacts live under root `outputs/`.

## Blog 1 Runtime

Blog 1 now uses τ³-Bench retail for the benchmark/environment. The teacher is served with vLLM raw completions so the notebook owns prompt rendering and Qwen XML parsing:

```bash
source /Users/kargarisaac/.venv-vllm-metal/bin/activate
HF_HUB_DISABLE_XET=1 VLLM_METAL_MEMORY_FRACTION=0.95 vllm serve mlx-community/Qwen3.5-35B-A3B-8bit \
  --host 127.0.0.1 \
  --port 8092 \
  --max-model-len 81920 \
  --dtype float16 \
  --trust-remote-code \
  --enforce-eager \
  --generation-config vllm
```

The vLLM-Metal server lives in its own Python 3.12 environment because the Apple Silicon Metal wheel is `cp312` only. The project `uv` environment remains the notebook/training environment.

The student SFT path uses MLX-LM LoRA on Apple Silicon:

```text
mlx-community/Qwen3.5-0.8B-MLX-bf16
```

The first MLX training config is intentionally conservative: the bf16 MLX conversion of Qwen 0.8B, one LoRA layer, batch size 1, prompt masking, and rows up to the verified local sequence length. The 4-bit MLX checkpoint is only a fallback for memory pressure, not the default blog path.
