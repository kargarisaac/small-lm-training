from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import generation
from common import sql_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model in the deterministic SQL-agent harness.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR / "sql_agent_bird_critic")
    parser.add_argument("--partition", choices=["train", "eval"], default="eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = sql_agent.load_rows(args.data_dir, args.partition, args.limit)
    output_path = args.output or cfg.OUTPUT_DIR / f"{cfg.filename_slug(args.model)}_sql_agent_{args.partition}_baml_eval.json"
    results = []
    if output_path.exists():
        results = cfg.make_json_safe(__import__("json").loads(output_path.read_text(encoding="utf-8")).get("results", []))
        print(f"Loaded {len(results)} cached results from: {output_path}", flush=True)
    done = {row["id"] for row in results}
    generate = generation.make_baml_generator(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
    )

    print("Harness LLM call path: BAML over OpenAI-compatible HTTP", flush=True)
    print("Model:", args.model, flush=True)
    print("Base URL:", args.base_url, flush=True)
    print("Rows:", len(rows), flush=True)
    print("Max turns:", args.max_turns, flush=True)
    for index, row in enumerate(rows, start=1):
        if row["id"] in done:
            print(f"{index}/{len(rows)} {row['id']} cached", flush=True)
            continue
        result = sql_agent.run_task(row, data_dir=args.data_dir, generate=generate, max_turns=args.max_turns)
        results.append(result)
        summary = sql_agent.summarize_results(results)
        cfg.write_json(output_path, {"summary": summary.__dict__, "results": results, "config": vars(args)})
        print(f"{index}/{len(rows)} {row['id']} success={result['success']} stop={result['stop_reason']} turns={result['turns']} running_success={summary.success}/{summary.total}", flush=True)

    summary = sql_agent.summarize_results(results)
    cfg.write_json(output_path, {"summary": summary.__dict__, "results": results, "config": vars(args)})
    print()
    print(f"Success: {summary.success}/{summary.total} = {summary.success_rate:.3f}")
    print(f"Submitted: {summary.submitted}/{summary.total}")
    print(f"Parse failures: {summary.parse_failures}")
    print(f"Max-turn failures: {summary.max_turn_failures}")
    print(f"Average turns: {summary.average_turns:.2f}")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
