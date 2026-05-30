from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import generation
from common import nestful


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on NESTFUL nested function-call sequence prediction.")
    parser.add_argument("--backend", choices=["hf", "mlx", "openai"], default="hf")
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR / "nestful")
    parser.add_argument("--partition", choices=["train", "eval"], default="eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.model is None:
        if args.backend == "hf":
            args.model = cfg.HF_STUDENT_MODEL
        elif args.backend == "mlx":
            args.model = cfg.MLX_STUDENT_MODEL
        else:
            raise ValueError("--model is required for --backend openai.")

    rows = nestful.load_prepared_rows(args.data_dir, args.partition, args.limit)
    generate = generation.make_generator(
        backend=args.backend,
        model_name=args.model,
        adapter=args.adapter,
        max_new_tokens=args.max_new_tokens,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
    )

    print("Backend:", args.backend)
    print("Model:", args.model)
    if args.backend == "openai":
        print("OpenAI-compatible base URL:", args.base_url)
        print("Reasoning effort:", args.reasoning_effort)
    print("Rows:", len(rows))

    results = []
    for index, row in enumerate(rows, start=1):
        result = nestful.score_output(row, generate(row))
        results.append(result)
        print(
            f"{index}/{len(rows)} {row['id']} "
            f"calls={len(row['expected_calls'])} parsed={result['predicted'] is not None} "
            f"names={result['name_sequence_correct']} exact={result['exact_correct']}"
        )

    summary = nestful.summarize_eval(results)
    output_path = args.output or cfg.OUTPUT_DIR / f"{cfg.filename_slug(args.model)}_nestful_{args.partition}_{args.backend}_eval.json"
    cfg.write_json(output_path, {"summary": summary.__dict__, "results": results})
    print()
    print(f"Exact accuracy: {summary.exact_correct}/{summary.total} = {summary.exact_accuracy:.3f}")
    print(f"Name-sequence accuracy: {summary.name_sequence_correct}/{summary.total} = {summary.name_sequence_accuracy:.3f}")
    print(f"Parse rate: {summary.parsed}/{summary.total} = {summary.parse_rate:.3f}")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
