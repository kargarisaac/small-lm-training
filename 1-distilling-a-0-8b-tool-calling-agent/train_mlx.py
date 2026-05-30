from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import nestful


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NESTFUL with MLX-LM LoRA.")
    parser.add_argument("--model", default=cfg.MLX_STUDENT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR / "nestful")
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--iters", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = cfg.read_jsonl(args.train_path, args.limit) if args.train_path else nestful.load_prepared_rows(args.data_dir, "train", args.limit)
    train_rows, valid_rows = nestful.split_train_validation(rows, args.validation_fraction)
    work_dir = args.output_dir or cfg.OUTPUT_DIR / f"{cfg.filename_slug(args.model)}_nestful_mlx"
    mlx_data_dir = work_dir / "mlx_lm_data"
    adapter_dir = work_dir / "adapter"
    nestful.write_mlx_lm_data(mlx_data_dir, train_rows, valid_rows)
    iters = args.iters if args.iters > 0 else math.ceil(len(train_rows) / (args.batch_size * args.grad_accum))
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--train",
        "--model",
        args.model,
        "--data",
        str(mlx_data_dir),
        "--adapter-path",
        str(adapter_dir),
        "--iters",
        str(iters),
        "--batch-size",
        str(args.batch_size),
        "--grad-accumulation-steps",
        str(args.grad_accum),
        "--learning-rate",
        str(args.learning_rate),
        "--num-layers",
        str(args.num_layers),
        "--max-seq-length",
        str(args.max_seq_length),
        "--mask-prompt",
        "--grad-checkpoint",
    ]
    print("Model:", args.model)
    print("Raw rows:", len(rows))
    print("Train rows:", len(train_rows))
    print("Validation rows:", len(valid_rows))
    print("Iters:", iters)
    print("Adapter dir:", adapter_dir)
    print("Command:", " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check=True)
    cfg.write_json(work_dir / "training_config.json", vars(args) | {"iters": iters, "adapter_dir": adapter_dir})


if __name__ == "__main__":
    main()
