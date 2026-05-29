from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import mlx_resumable_lora, sft_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tau3 retail student with MLX-LM LoRA.")
    parser.add_argument("--student-model", default=cfg.STUDENT_MODEL)
    parser.add_argument("--teacher-model", default=cfg.TEACHER_MODEL)
    parser.add_argument("--teacher-provider", default="vllm_raw")
    parser.add_argument("--sft-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-seq-length", type=int, default=16500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument("--lora-layers", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 means one pass over kept rows.")
    parser.add_argument("--resume", default="none", help="Use 'latest', 'none', or a checkpoint path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = cfg.setup_notebook_paths(blog_dir_name="1-distilling-a-0-8b-tool-calling-agent")
    user_model = cfg.required_env("TAU_BENCH_USER_SIMULATOR_LLM")
    user_slug = cfg.filename_slug(user_model)
    teacher_slug = cfg.filename_slug(args.teacher_model)
    student_slug = cfg.filename_slug(args.student_model)
    sft_path = Path(args.sft_path) if args.sft_path else (
        paths.output_dir / f"{teacher_slug}_{args.teacher_provider}_tau3_bench_retail_train_successful_sft_chat_rows_{user_slug}.jsonl"
    )
    work_dir = Path(args.output_dir) if args.output_dir else (
        paths.output_dir / f"{student_slug}_tau3_retail_sft_mlx_lm"
    )
    data_dir = work_dir / "mlx_lm_data"
    adapter_dir = work_dir / f"{student_slug}_tau3_retail_mlx_lora_adapter"

    if not sft_path.exists():
        raise FileNotFoundError(f"SFT rows file not found: {sft_path}")

    tokenizer = cfg.load_tokenizer(args.student_model)
    rows = cfg.load_jsonl(sft_path)
    lengths = [sft_rows.mlx_chat_row_token_length(row, tokenizer) for row in rows]
    rows_that_fit = [row for row, length in zip(rows, lengths) if length <= args.max_seq_length]
    rows_too_long = len(rows) - len(rows_that_fit)
    if not rows_that_fit:
        raise RuntimeError("No SFT rows fit max sequence length.")

    train_rows, validation_rows, validation_task_ids = sft_rows.split_sft_rows_by_task_id(
        rows_that_fit,
        validation_task_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    if not validation_rows:
        validation_rows = train_rows[:1]

    data_dir.mkdir(parents=True, exist_ok=True)
    cfg.write_jsonl(data_dir / "train.jsonl", train_rows)
    cfg.write_jsonl(data_dir / "valid.jsonl", validation_rows)
    cfg.write_jsonl(data_dir / "test.jsonl", validation_rows)

    max_steps = args.max_steps
    if max_steps <= 0:
        max_steps = max(1, len(train_rows) // max(1, args.batch_size))
    save_steps = max(1, min(25, max_steps))
    eval_steps = max(1, min(25, max_steps))
    training_config = mlx_resumable_lora.ResumableLoraConfig(
        model=args.student_model,
        data_dir=data_dir,
        adapter_path=adapter_dir,
        total_iters=max_steps,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        val_batches=1,
        steps_per_report=1,
        steps_per_eval=eval_steps,
        save_every=save_steps,
        num_layers=args.lora_layers,
        lora_rank=args.lora_rank,
        lora_scale=args.lora_scale,
        mask_prompt=True,
        seed=args.training_seed,
        grad_checkpoint=True,
        clear_cache_threshold=0,
        resume=args.resume,
    )

    print("Student model:", args.student_model)
    print("SFT rows:", sft_path)
    print("Rows kept:", len(rows_that_fit), "dropped:", rows_too_long)
    print("Train rows:", len(train_rows), "validation rows:", len(validation_rows))
    print("Validation task ids:", sorted(validation_task_ids))
    print("Max steps:", max_steps)
    print("Batch size:", args.batch_size)
    print("LoRA layers:", args.lora_layers)
    print("LoRA rank:", args.lora_rank)
    print("LoRA scale:", args.lora_scale)
    print("Adapter dir:", adapter_dir)
    print("Resume:", args.resume)

    training_result = mlx_resumable_lora.run_resumable_lora_training(training_config)
    metadata = {
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "teacher_provider": args.teacher_provider,
        "sft_path": str(sft_path),
        "mlx_data_dir": str(data_dir),
        "adapter_dir": str(adapter_dir),
        "raw_rows": len(rows),
        "rows_kept": len(rows_that_fit),
        "rows_dropped_over_length": rows_too_long,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "max_steps": max_steps,
        "lora_layers": args.lora_layers,
        "lora_rank": args.lora_rank,
        "lora_scale": args.lora_scale,
        "resume": args.resume,
        "training_result": training_result,
    }
    metadata_path = work_dir / "training_metadata_mlx_lm.json"
    metadata_path.write_text(json.dumps(cfg.make_json_safe(metadata), indent=2), encoding="utf-8")
    print("Saved adapter:", adapter_dir)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
