from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import trace_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize tau3-bench retail eval traces.")
    parser.add_argument("--student-model", default="Qwen/Qwen3.5-0.8B")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = cfg.setup_notebook_paths(blog_dir_name="1-distilling-a-0-8b-tool-calling-agent")
    user_model = cfg.required_env("TAU_BENCH_USER_SIMULATOR_LLM")
    bundle = trace_stats.load_tau_bench_eval_trace_stats(
        output_dir=paths.output_dir,
        user_simulator_model=user_model,
        student_model=args.student_model,
    )
    print("Result files:")
    for label, path in bundle["result_files"]:
        print(f"- {label}: {path}")
    if bundle["stats_df"].empty:
        print("No eval result files found.")
        return

    print()
    print(bundle["summary_df"].round(3).to_string())
    stats_csv_path = trace_stats.save_tau_bench_eval_trace_stats_csv(bundle["stats_df"], paths.output_dir)
    print()
    print("Saved per-task stats:", stats_csv_path)


if __name__ == "__main__":
    main()
