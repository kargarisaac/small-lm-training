from __future__ import annotations

import argparse
import math
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import sql_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SQL-agent student with MLX-LM LoRA.")
    parser.add_argument("--model", default=cfg.MLX_STUDENT_MODEL)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=cfg.SFT_MAX_SEQ_LENGTH)
    parser.add_argument("--batch-size", type=int, default=cfg.SFT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=cfg.SFT_GRAD_ACCUM)
    parser.add_argument("--learning-rate", type=float, default=cfg.SFT_LEARNING_RATE)
    parser.add_argument("--lora-rank", type=int, default=cfg.SFT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=float, default=cfg.SFT_LORA_ALPHA)
    parser.add_argument("--num-layers", type=int, default=cfg.SFT_MLX_NUM_LAYERS)
    parser.add_argument("--resume-adapter-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=cfg.SFT_SEED)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--steps-per-eval", type=int, default=200)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=cfg.SFT_VALIDATION_FRACTION)
    parser.add_argument("--iters", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.train_path is None:
        raise ValueError("--train-path is required. Run notebook 02 to write the final filtered SFT file first.")
    rows = cfg.read_jsonl(args.train_path, args.limit)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prepared = sql_agent.prepare_sft_rows(rows, tokenizer, args.max_seq_length, args.validation_fraction)
    train_rows = prepared["train_rows"]
    valid_rows = prepared["valid_rows"]
    work_dir = args.output_dir or cfg.OUTPUT_DIR / f"{cfg.filename_slug(args.model)}_sql_agent_mlx"
    mlx_data_dir = work_dir / "mlx_lm_data"
    adapter_dir = work_dir / "adapter"
    if not train_rows:
        raise RuntimeError("No train rows fit max sequence length.")
    if not valid_rows:
        valid_rows = train_rows[:1]
    iters = args.iters if args.iters > 0 else math.ceil(len(train_rows) / (args.batch_size * args.grad_accum))
    from mlx_lm import lora

    lora_args = dict(lora.CONFIG_DEFAULTS)
    lora_args.update(
        train=True,
        model=args.model,
        data=str(mlx_data_dir),
        adapter_path=str(adapter_dir),
        iters=iters,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_layers=args.num_layers,
        resume_adapter_file=str(args.resume_adapter_file) if args.resume_adapter_file else None,
        seed=args.seed,
        save_every=args.save_every,
        steps_per_eval=args.steps_per_eval,
        steps_per_report=args.steps_per_report,
        max_seq_length=args.max_seq_length,
        lora_parameters={"rank": args.lora_rank, "dropout": 0.0, "scale": args.lora_alpha},
        mask_prompt=True,
        grad_checkpoint=True,
    )
    print("Model:", args.model)
    for key, value in prepared["stats"].items():
        print(f"{key.replace('_', ' ').title()}:", value)
    print("Iters:", iters)
    print("Adapter dir:", adapter_dir)
    print("MLX-LM LoRA args:", {key: lora_args[key] for key in ("model", "data", "adapter_path", "iters", "batch_size", "grad_accumulation_steps", "learning_rate", "num_layers", "max_seq_length", "lora_parameters", "mask_prompt", "resume_adapter_file", "seed", "save_every")})
    if "qwen" in args.model.lower() and not cfg.QWEN_ENABLE_THINKING:
        print("Note: direct Qwen inference and SFT token-length checks use enable_thinking=False.")
        print("Note: mlx_lm.lora does not expose chat-template kwargs, so prefer train_unsloth.py for exact no-thinking SFT.")
    if args.dry_run:
        return
    sql_agent.write_mlx_lm_data(mlx_data_dir, train_rows, valid_rows)
    lora.run(types.SimpleNamespace(**lora_args))
    cfg.write_json(work_dir / "training_config.json", vars(args) | {"iters": iters, "adapter_dir": adapter_dir, "stats": prepared["stats"]})


if __name__ == "__main__":
    main()
