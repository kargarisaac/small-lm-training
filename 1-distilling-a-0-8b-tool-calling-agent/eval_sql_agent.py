from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import generation
from common import sql_agent


def wait_for_openai_server(base_url: str, wait_seconds: int) -> dict[str, Any]:
    models_url = f"{base_url.rstrip('/')}/models"
    deadline = time.monotonic() + wait_seconds
    last_error: Exception | None = None
    while True:
        try:
            response = httpx.get(models_url, timeout=5)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"Unexpected /models response: {data!r}")
            return data
        except Exception as error:
            last_error = error
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"No OpenAI-compatible server is reachable at {models_url}. "
                    "Start vLLM/MLX and wait until /v1/models responds before running eval."
                ) from last_error
            time.sleep(2)


def model_context_length(models_response: dict[str, Any], model: str) -> int | None:
    data = models_response.get("data")
    if not isinstance(data, list):
        return None
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("id") == model or item.get("root") == model:
            value = item.get("max_model_len")
            return int(value) if isinstance(value, int) else None
    return None


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
    parser.add_argument("--server-wait-seconds", type=int, default=180)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    models_response = wait_for_openai_server(args.base_url, args.server_wait_seconds)
    max_model_len = model_context_length(models_response, args.model)
    if max_model_len is not None and args.max_new_tokens >= max_model_len:
        raise ValueError(
            f"--max-new-tokens={args.max_new_tokens} is not usable with server max_model_len={max_model_len}; "
            "it leaves no room for the prompt. Increase vLLM --max-model-len or lower --max-new-tokens."
        )

    rows = sql_agent.load_rows(args.data_dir, args.partition, args.limit)
    output_path = args.output or cfg.OUTPUT_DIR / f"{cfg.filename_slug(args.model)}_sql_agent_{args.partition}_baml_eval.json"
    results = []
    if output_path.exists():
        results = cfg.make_json_safe(json.loads(output_path.read_text(encoding="utf-8")).get("results", []))
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
        try:
            result = sql_agent.run_task(row, data_dir=args.data_dir, generate=generate, max_turns=args.max_turns)
        except Exception as error:
            result = {
                "id": row["id"],
                "db_id": row["db_id"],
                "category": row["category"],
                "success": False,
                "stop_reason": "runtime_error",
                "turns": 0,
                "trace": [
                    {
                        "turn": 0,
                        "stop_reason": "runtime_error",
                        "error_type": type(error).__name__,
                        "error": str(error)[:4000],
                    }
                ],
            }
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
    print(f"Repeated-action failures: {summary.repeated_action_failures}")
    print(f"Runtime errors: {summary.runtime_errors}")
    print(f"Average turns: {summary.average_turns:.2f}")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
