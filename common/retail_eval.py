from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from uuid import uuid4
import json
import time
import threading

from .agents import TauBenchRetailMlxStudentAgent, TauBenchRetailQwenRawCompletionAgent
from .config import (
    MlflowConfig,
    TauBenchRetailEvalConfig,
    TeacherConfig,
    _configure_litellm_for_notebooks,
    _quiet_external_logs,
    make_json_safe,
)
from .mlflow_logging import TauBenchRetailMlflowLogger
from .tau_runtime import TauBenchRetailRuntime


def _retail_user_tools(runtime: TauBenchRetailRuntime, environment: Any, task: Any) -> list[Any] | None:
    try:
        return environment.get_user_tools(include=task.user_tools) or None
    except Exception:
        return None


def _simulation_row(task: Any, simulation: Any, started: float) -> dict[str, Any]:
    reward = simulation.reward_info.reward if simulation.reward_info is not None else 0.0
    return {
        "task_id": str(task.id),
        "reward": reward,
        "is_success": reward == 1.0,
        "termination_reason": simulation.termination_reason,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "message_count": len(simulation.messages or []),
        "reward_info": make_json_safe(simulation.reward_info),
        "simulation": make_json_safe(simulation),
        "error": None,
    }


def _exception_row(task: Any, started: float, exc: Exception) -> dict[str, Any]:
    return {
        "task_id": str(task.id),
        "reward": 0.0,
        "is_success": False,
        "termination_reason": "exception",
        "duration_seconds": round(time.perf_counter() - started, 3),
        "message_count": 0,
        "reward_info": None,
        "simulation": None,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def _run_task_with_agent(
    *,
    runtime: TauBenchRetailRuntime,
    task: Any,
    task_index: int,
    config: TauBenchRetailEvalConfig,
    trace_dir: Path,
    simulation_prefix: str,
    make_agent: Callable[[Any], Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        environment = runtime.get_environment()
        user = runtime.UserSimulator(
            llm=config.user_simulator_model,
            instructions=str(task.user_scenario),
            tools=_retail_user_tools(runtime, environment, task),
            llm_args=config.user_simulator_args or {},
        )
        orchestrator = runtime.Orchestrator(
            domain="retail",
            agent=make_agent(environment),
            user=user,
            environment=environment,
            task=task,
            max_steps=config.max_steps,
            max_errors=config.max_errors,
            seed=config.seed + task_index,
            simulation_id=f"{simulation_prefix}_{task.id}_{uuid4().hex[:8]}",
            validate_communication=True,
        )
        simulation = runtime.run_simulation(orchestrator, evaluation_type=runtime.EvaluationType.ALL)
        row = _simulation_row(task, simulation, started)
    except Exception as exc:
        row = _exception_row(task, started, exc)

    trace_path = trace_dir / f"{row['task_id']}.json"
    trace_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


class TauBenchRetailMlxStudentEvalRunner:
    def __init__(
        self,
        *,
        runtime: TauBenchRetailRuntime,
        model: Any,
        tokenizer: Any,
        qwen_tools: list[dict[str, Any]],
        tool_schema_by_name: dict[str, dict[str, Any]],
        sampler: Any,
        config: TauBenchRetailEvalConfig,
        trace_dir: Path,
        generation_lock: threading.Lock | None = None,
    ):
        self.runtime = runtime
        self.model = model
        self.tokenizer = tokenizer
        self.qwen_tools = qwen_tools
        self.tool_schema_by_name = tool_schema_by_name
        self.sampler = sampler
        self.config = config
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.generation_lock = generation_lock

    def run_task(self, task: Any, task_index: int) -> dict[str, Any]:
        return _run_task_with_agent(
            runtime=self.runtime,
            task=task,
            task_index=task_index,
            config=self.config,
            trace_dir=self.trace_dir,
            simulation_prefix="mlx_student_tau3_bench_retail",
            make_agent=lambda environment: TauBenchRetailMlxStudentAgent(
                runtime=self.runtime,
                model=self.model,
                tokenizer=self.tokenizer,
                qwen_tools=self.qwen_tools,
                tool_schema_by_name=self.tool_schema_by_name,
                domain_policy=environment.get_policy(),
                max_new_tokens=self.config.max_new_tokens,
                sampler=self.sampler,
                generation_lock=self.generation_lock,
            ),
        )


class TauBenchRetailTeacherEvalRunner:
    def __init__(
        self,
        *,
        runtime: TauBenchRetailRuntime,
        tokenizer: Any,
        qwen_tools: list[dict[str, Any]],
        tool_schema_by_name: dict[str, dict[str, Any]],
        teacher_config: TeacherConfig,
        config: TauBenchRetailEvalConfig,
        trace_dir: Path,
    ):
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.qwen_tools = qwen_tools
        self.tool_schema_by_name = tool_schema_by_name
        self.teacher_config = teacher_config
        self.config = config
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def run_task(self, task: Any, task_index: int) -> dict[str, Any]:
        return _run_task_with_agent(
            runtime=self.runtime,
            task=task,
            task_index=task_index,
            config=self.config,
            trace_dir=self.trace_dir,
            simulation_prefix="teacher_tau3_bench_retail",
            make_agent=lambda environment: TauBenchRetailQwenRawCompletionAgent(
                runtime=self.runtime,
                tokenizer=self.tokenizer,
                qwen_tools=self.qwen_tools,
                tool_schema_by_name=self.tool_schema_by_name,
                domain_policy=environment.get_policy(),
                teacher_config=self.teacher_config,
            ),
        )


def check_tau_bench_retail_user_simulator(
    *,
    runtime: TauBenchRetailRuntime,
    task: Any,
    user_simulator_model: str,
    user_simulator_args: dict[str, Any] | None,
) -> dict[str, Any]:
    _configure_litellm_for_notebooks()
    with _quiet_external_logs():
        environment = runtime.get_environment()
        try:
            user_tools = environment.get_user_tools(include=task.user_tools) or None
        except Exception:
            user_tools = None

        user = runtime.UserSimulator(
            llm=user_simulator_model,
            instructions=str(task.user_scenario),
            tools=user_tools,
            llm_args=user_simulator_args or {},
        )
        state = user.get_init_state()
        message, state = user.generate_next_message(runtime.DEFAULT_FIRST_AGENT_MESSAGE, state)

    has_visible_user_reply = bool(message.content and message.content.strip())
    has_user_tool_call = bool(message.tool_calls)
    if not has_visible_user_reply and not has_user_tool_call:
        raise RuntimeError("User simulator returned neither visible text nor a user tool call.")

    return {
        "task_id": str(task.id),
        "model": user_simulator_model,
        "user_side_tool_count": len(user_tools or []),
        "message": make_json_safe(message),
        "state": make_json_safe(state),
        "content": message.content,
        "tool_calls": make_json_safe(message.tool_calls or []),
    }


def _summarize_eval_payload(
    *,
    task_objects: list[Any],
    rows_by_id: dict[str, dict[str, Any]],
    config: TauBenchRetailEvalConfig,
    output_path: Path,
) -> dict[str, Any]:
    ordered_rows = [
        rows_by_id[str(task.id)]
        for task in task_objects
        if str(task.id) in rows_by_id
    ]
    correct = sum(1 for item in ordered_rows if item["is_success"])
    return {
        "benchmark": "τ³-Bench retail test",
        "dataset_revision": config.dataset_revision,
        "model_role": config.model_role,
        "agent_model": config.student_model_name,
        "student_model": config.student_model_name,
        "user_simulator_model": config.user_simulator_model,
        "user_simulator_args": config.user_simulator_args or {},
        "pass_at": config.pass_at,
        "task_count": len(task_objects),
        "completed_count": len(ordered_rows),
        "correct_count": correct,
        "accuracy": correct / len(ordered_rows) if ordered_rows else 0.0,
        "max_steps": config.max_steps,
        "max_errors": config.max_errors,
        "max_new_tokens": config.max_new_tokens,
        "output_path": str(output_path),
        "task_results": ordered_rows,
    }


def run_tau_bench_retail_eval_tasks(
    *,
    task_objects: list[Any],
    runner: Any,
    output_path: Path,
    print_progress: bool = True,
    show_progress_bar: bool = True,
    quiet_tau2_console: bool = True,
    mlflow_config: MlflowConfig | None = None,
    mlflow_run_name: str | None = None,
    mlflow_tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    cached_rows_by_id: dict[str, dict[str, Any]] = {}
    model_role = getattr(runner.config, "model_role", "student")
    if output_path.exists():
        cached_payload = json.loads(output_path.read_text(encoding="utf-8"))
        requested_task_ids = {str(task.id) for task in task_objects}
        cached_rows_by_id = {
            row["task_id"]: row
            for row in cached_payload.get("task_results", [])
            if row.get("task_id") in requested_task_ids
        }
        if print_progress:
            print(f"Loaded {len(cached_rows_by_id)} cached {model_role} eval results from: {output_path}")

    pending_tasks = [
        (index, task)
        for index, task in enumerate(task_objects)
        if str(task.id) not in cached_rows_by_id
    ]

    if print_progress:
        print("Cached tasks:", len(cached_rows_by_id))
        print("Pending tasks:", len(pending_tasks))

    completed_rows = dict(cached_rows_by_id)
    correct_count = sum(1 for row in completed_rows.values() if row.get("is_success"))
    completed_count = len(completed_rows)

    progress_bar = None
    if print_progress and show_progress_bar:
        try:
            from tqdm.auto import tqdm  # noqa: PLC0415

            progress_bar = tqdm(
                total=len(task_objects),
                initial=completed_count,
                desc=f"{model_role} eval",
                unit="task",
                dynamic_ncols=True,
                leave=True,
            )
        except ImportError:
            progress_bar = None

    if progress_bar is not None:
        accuracy = correct_count / completed_count if completed_count else 0.0
        progress_bar.set_postfix(
            {"valid": f"{correct_count}/{completed_count}", "acc": f"{accuracy:.1%}"}
        )

    run_name = mlflow_run_name or output_path.stem
    with TauBenchRetailMlflowLogger(
        config=mlflow_config,
        run_name=run_name,
        eval_config=runner.config,
        output_path=output_path,
        trace_dir=getattr(runner, "trace_dir", None),
        total_tasks=len(task_objects),
        tags=mlflow_tags,
    ) as mlflow_logger:
        for step, task in enumerate(task_objects, start=1):
            task_id = str(task.id)
            if task_id in cached_rows_by_id:
                mlflow_logger.log_task(
                    completed_rows[task_id],
                    step=step,
                    completed_count=sum(
                        1
                        for previous_task in task_objects[:step]
                        if str(previous_task.id) in completed_rows
                    ),
                    correct_count=sum(
                        1
                        for previous_task in task_objects[:step]
                        if str(previous_task.id) in completed_rows
                        and completed_rows[str(previous_task.id)].get("is_success")
                    ),
                )
                continue

            with _quiet_external_logs(quiet_tau2_console):
                row = runner.run_task(task, step - 1)
            completed_rows[task_id] = row
            correct_count = sum(1 for item in completed_rows.values() if item.get("is_success"))
            completed_count = len(completed_rows)
            status = "valid" if row["is_success"] else row["termination_reason"]
            if row.get("error"):
                status = f"error:{row['error']['type']}"
            if progress_bar is not None:
                progress_bar.update(1)
                accuracy = correct_count / completed_count if completed_count else 0.0
                progress_bar.set_postfix(
                    {
                        "valid": f"{correct_count}/{completed_count}",
                        "acc": f"{accuracy:.1%}",
                        "last": task_id,
                        "status": status[:40],
                    }
                )
            elif print_progress:
                print(
                    f"{task_id}: valid={correct_count}/{completed_count} "
                    f"reward={row['reward']} status={status} "
                    f"duration={row['duration_seconds']}s"
                )

            mlflow_logger.log_task(
                row,
                step=step,
                completed_count=completed_count,
                correct_count=correct_count,
            )

            payload = _summarize_eval_payload(
                task_objects=task_objects,
                rows_by_id=completed_rows,
                config=runner.config,
                output_path=output_path,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        if not pending_tasks and print_progress:
            if progress_bar is None:
                print("No pending tasks. Using cached results.")

        if progress_bar is not None:
            progress_bar.close()

        payload = _summarize_eval_payload(
            task_objects=task_objects,
            rows_by_id=completed_rows,
            config=runner.config,
            output_path=output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        mlflow_logger.log_summary(payload)

    return json.loads(output_path.read_text(encoding="utf-8"))
