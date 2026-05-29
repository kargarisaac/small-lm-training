from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import retail_eval, tau_runtime, user_simulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an MLX-LM student on tau3-bench retail.")
    parser.add_argument("--model", default=cfg.STUDENT_MODEL)
    parser.add_argument("--adapter", default=None, help="Optional MLX adapter directory.")
    parser.add_argument("--limit", type=int, default=0, help="0 means all test tasks.")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=cfg.MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = cfg.setup_notebook_paths(blog_dir_name="1-distilling-a-0-8b-tool-calling-agent")
    user_simulator_runtime = user_simulator.start_tau_bench_user_simulator_from_env()
    user_model = user_simulator_runtime.model
    user_args = user_simulator_runtime.args

    from mlx_lm import load as mlx_load
    from mlx_lm.sample_utils import make_sampler

    adapter_path = str(args.adapter) if args.adapter else None
    model, tokenizer = mlx_load(args.model, adapter_path=adapter_path)
    sampler = make_sampler(temp=0.0, top_p=1.0, top_k=0)

    retail_domain = tau_runtime.load_tau_bench_retail_domain(paths.data_dir, cfg.TAU_BENCH_REPO_REVISION)
    tasks = retail_domain.runtime.get_tasks("test")
    task_objects = tasks[: args.limit] if args.limit > 0 else tasks

    user_slug = cfg.filename_slug(user_model)
    model_slug = cfg.filename_slug(args.model)
    if args.adapter:
        model_slug = f"{model_slug}_{cfg.filename_slug(Path(args.adapter).name)}"
    run_label = "mlx_sft_student" if args.adapter else "mlx_student"
    output_path = paths.output_dir / f"{model_slug}_tau3_bench_retail_test_official_{run_label}_eval_{user_slug}.json"
    trace_dir = paths.output_dir / "local_traces" / output_path.stem
    mlflow_config = cfg.MlflowConfig(
        enabled=args.mlflow,
        experiment_name="distillation-blogs-tau3",
        log_full_artifacts=True,
        log_spans=False,
    )
    eval_config = cfg.TauBenchRetailEvalConfig(
        dataset_revision=cfg.TAU_BENCH_REPO_REVISION,
        student_model_name=f"{args.model} + {args.adapter}" if args.adapter else args.model,
        user_simulator_model=user_model,
        user_simulator_args=user_args,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        model_role=run_label,
    )
    runner = retail_eval.TauBenchRetailMlxStudentEvalRunner(
        runtime=retail_domain.runtime,
        model=model,
        tokenizer=tokenizer,
        qwen_tools=retail_domain.tools,
        tool_schema_by_name=retail_domain.tool_schema_by_name,
        sampler=sampler,
        config=eval_config,
        trace_dir=trace_dir,
    )

    print("Student model:", args.model)
    print("Adapter:", args.adapter or "<none>")
    print("User simulator:", user_model)
    print("Tasks:", len(task_objects))
    print("Output:", output_path)
    payload = retail_eval.run_tau_bench_retail_eval_tasks(
        task_objects=task_objects,
        runner=runner,
        output_path=output_path,
        print_progress=True,
        show_progress_bar=True,
        quiet_tau2_console=True,
        mlflow_config=mlflow_config,
        mlflow_run_name=output_path.stem,
        mlflow_tags={"tau3.script": Path(__file__).name, "tau3.runtime": "mlx_lm"},
    )
    rows = payload["task_results"]
    correct = sum(1 for row in rows if row["is_success"])
    failures = Counter(row["termination_reason"] for row in rows if not row["is_success"])
    print()
    print(f"Accuracy: {correct}/{len(rows)} = {correct / len(rows):.3f}" if rows else "No rows.")
    print("Failures:", json.dumps(dict(failures), indent=2))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
