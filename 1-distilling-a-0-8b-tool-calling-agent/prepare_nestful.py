from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import nestful


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NESTFUL JSONL files.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-fraction", type=float, default=cfg.NESTFUL_EVAL_FRACTION)
    parser.add_argument("--seed", type=int, default=cfg.NESTFUL_SPLIT_SEED)
    args = parser.parse_args()

    output_dir = args.output_dir or cfg.DATA_DIR / (f"nestful_calls_le_{args.max_calls}" if args.max_calls is not None else "nestful")
    stats = nestful.prepare_data(output_dir, args.train_limit, args.eval_limit, args.eval_fraction, args.seed, args.max_calls)
    print("Prepared NESTFUL data in:", output_dir)
    print("Max calls filter:", stats["max_calls"])
    print("Train rows:", stats["train_rows"])
    print("Eval rows:", stats["eval_rows"])
    print("Sequence length:", stats["sequence_length"])
    print("Tool count:", stats["tool_count"])
    print("Top called functions:", stats["top_called_functions"])


if __name__ == "__main__":
    main()
