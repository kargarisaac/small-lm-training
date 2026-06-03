from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import generation
from common import sql_agent


def run_teacher_task(row: dict, args: argparse.Namespace) -> dict:
    generate = generation.make_baml_generator(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
    )
    return sql_agent.run_task(row, data_dir=args.data_dir, generate=generate, max_turns=args.max_turns, keep_messages=True)


def worker(queue: mp.Queue, row: dict, args: argparse.Namespace) -> None:
    try:
        queue.put(run_teacher_task(row, args))
    except Exception as error:
        queue.put(runtime_error_result(row, error))


def run_openai_task_with_timeout(row: dict, args: argparse.Namespace) -> dict:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=worker, args=(queue, row, args))
    process.start()
    process.join(args.task_timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(10)
        return runtime_error_result(row, TimeoutError(f"task timed out after {args.task_timeout_seconds}s"))
    if queue.empty():
        return runtime_error_result(row, RuntimeError(f"worker exited with code {process.exitcode} without a result"))
    return queue.get()


def runtime_error_result(row: dict, error: BaseException) -> dict:
    return {
        "id": row["id"],
        "db_id": row["db_id"],
        "category": row["category"],
        "success": False,
        "stop_reason": "teacher_runtime_error",
        "turns": 0,
        "trace": [{"error": f"{type(error).__name__}: {error}"}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a teacher through the SQL-agent harness and keep BAML-canonical SFT trace rows from successful trajectories.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR / "sql_agent_bird_critic")
    parser.add_argument("--partition", choices=["train", "eval"], default="train")
    parser.add_argument("--ids-path", type=Path, default=None, help="Optional newline or JSON-list file of exact task ids to run.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--task-timeout-seconds", type=int, default=600)
    parser.add_argument("--no-task-isolation", action="store_true")
    parser.add_argument("--retry-runtime-errors", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = sql_agent.load_rows(args.data_dir, args.partition, args.limit)
    if args.ids_path is not None:
        raw_ids = args.ids_path.read_text(encoding="utf-8").strip()
        selected_ids = {str(task_id).strip() for task_id in (json.loads(raw_ids) if raw_ids.startswith("[") else raw_ids.splitlines()) if str(task_id).strip()}
        rows = [row for row in rows if row["id"] in selected_ids]
        missing_ids = selected_ids - {row["id"] for row in rows}
        if missing_ids:
            raise ValueError(f"{args.ids_path} contains ids not found in {args.partition}: {sorted(missing_ids)[:20]}")
    print("Harness LLM call path: BAML over OpenAI-compatible HTTP", flush=True)
    output_path = args.output or cfg.OUTPUT_DIR / f"{cfg.filename_slug(args.model)}_sql_agent_{args.partition}_baml_sft_trace_rows.jsonl"
    report_path = output_path.with_suffix(".report.json")
    results = []
    sft_trace_rows = []
    if report_path.exists():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        results = cached.get("results", [])
        if output_path.exists():
            sft_trace_rows = cfg.read_jsonl(output_path)
        if args.retry_runtime_errors:
            retry_ids = {row["id"] for row in results if row.get("stop_reason") == "teacher_runtime_error"}
            results = [row for row in results if row["id"] not in retry_ids]
            print(f"Retrying {len(retry_ids)} cached teacher_runtime_error tasks.", flush=True)
        print(f"Loaded {len(results)} cached teacher results from: {report_path}", flush=True)
    done = {row["id"] for row in results}
    generate = None
    if args.no_task_isolation:
        generate = generation.make_baml_generator(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
        )
    for index, row in enumerate(rows, start=1):
        if row["id"] in done:
            print(f"{index}/{len(rows)} {row['id']} cached", flush=True)
            continue
        try:
            if not args.no_task_isolation:
                result = run_openai_task_with_timeout(row, args)
            else:
                result = sql_agent.run_task_with_timeout(
                    row,
                    data_dir=args.data_dir,
                    generate=generate,
                    max_turns=args.max_turns,
                    timeout_seconds=args.task_timeout_seconds,
                    keep_messages=True,
                )
        except Exception as error:
            result = runtime_error_result(row, error)
        results.append(result)
        kept = sql_agent.successful_sft_trace_rows(result, args.model, "baml_openai_compatible")
        sft_trace_rows.extend(kept)
        summary = sql_agent.summarize_results(results)
        cfg.write_jsonl(output_path, sft_trace_rows)
        cfg.write_json(report_path, {"summary": summary.__dict__, "results": results, "sft_trace_rows": len(sft_trace_rows), "config": vars(args)})
        print(f"{index}/{len(rows)} {row['id']} success={result['success']} stop={result['stop_reason']} turns={result['turns']} running_success={summary.success}/{summary.total} kept_sft_trace_rows={len(sft_trace_rows)}", flush=True)

    summary = sql_agent.summarize_results(results)
    cfg.write_jsonl(output_path, sft_trace_rows)
    cfg.write_json(report_path, {"summary": summary.__dict__, "results": results, "sft_trace_rows": len(sft_trace_rows), "config": vars(args)})
    print()
    print(f"Teacher success: {summary.success}/{summary.total} = {summary.success_rate:.3f}")
    print("BAML-canonical SFT trace rows:", len(sft_trace_rows))
    print("Saved BAML-canonical SFT trace rows:", output_path)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
