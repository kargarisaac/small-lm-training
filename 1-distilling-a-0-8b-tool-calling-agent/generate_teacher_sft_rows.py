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
    parser = argparse.ArgumentParser(description="Generate teacher SFT rows for NESTFUL.")
    parser.add_argument("--backend", choices=["hf", "mlx", "openai"], default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR / "nestful")
    parser.add_argument("--partition", choices=["train", "eval"], default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

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
    results = []
    sft_rows = []
    for index, row in enumerate(rows, start=1):
        result = nestful.score_output(row, generate(row))
        results.append(result)
        if result["exact_correct"]:
            sft_rows.append(nestful.teacher_sft_row(row, result, args.model, args.backend))
        print(
            f"{index}/{len(rows)} {row['id']} "
            f"parsed={result['predicted'] is not None} names={result['name_sequence_correct']} "
            f"exact={result['exact_correct']} kept={len(sft_rows)}"
        )

    summary = nestful.summarize_eval(results)
    output_path = args.output or cfg.OUTPUT_DIR / f"{cfg.filename_slug(args.model)}_nestful_{args.partition}_{args.backend}_teacher_sft_rows.jsonl"
    report_path = output_path.with_suffix(".report.json")
    cfg.write_jsonl(output_path, sft_rows)
    cfg.write_json(report_path, {"summary": summary.__dict__, "results": results, "sft_rows": len(sft_rows)})
    print()
    print(f"Teacher exact accuracy: {summary.exact_correct}/{summary.total} = {summary.exact_accuracy:.3f}")
    print(f"Teacher name-sequence accuracy: {summary.name_sequence_correct}/{summary.total} = {summary.name_sequence_accuracy:.3f}")
    print("Saved SFT rows:", output_path)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
