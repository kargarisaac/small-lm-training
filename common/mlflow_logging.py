from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .config import MlflowConfig, make_json_safe


def _setup_mlflow(config: MlflowConfig):
    import mlflow

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    return mlflow


def _safe_artifact_name(text: str) -> str:
    characters = [
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in text
    ]
    return "".join(characters).strip("._") or "artifact"


class TauBenchRetailMlflowLogger:
    def __init__(
        self,
        *,
        config: MlflowConfig | None,
        run_name: str,
        eval_config: TauBenchRetailEvalConfig,
        output_path: Path,
        trace_dir: Path | None,
        total_tasks: int,
        tags: dict[str, str] | None = None,
    ):
        self.config = config
        self.run_name = run_name
        self.eval_config = eval_config
        self.output_path = output_path
        self.trace_dir = trace_dir
        self.total_tasks = total_tasks
        self.tags = tags or {}
        self.mlflow = None
        self.active_run = None
        self.reported_failure = False

    def __enter__(self):
        if self.config is None or not self.config.enabled:
            return self

        try:
            self.mlflow = _setup_mlflow(self.config)
            self.active_run = self.mlflow.start_run(run_name=self.run_name)
            self.active_run.__enter__()
            self.mlflow.set_tags(
                {
                    "tau3.kind": "retail_eval",
                    "tau3.model_role": self.eval_config.model_role,
                    **self.tags,
                }
            )
            self._log_param("dataset_revision", self.eval_config.dataset_revision)
            self._log_param("agent_model", self.eval_config.student_model_name)
            self._log_param("user_simulator_model", self.eval_config.user_simulator_model)
            self._log_param("user_simulator_args", self.eval_config.user_simulator_args or {})
            self._log_param("max_steps", self.eval_config.max_steps)
            self._log_param("max_errors", self.eval_config.max_errors)
            self._log_param("max_new_tokens", self.eval_config.max_new_tokens)
            self._log_param("seed", self.eval_config.seed)
            self._log_param("output_path", str(self.output_path))
            if self.trace_dir is not None:
                self._log_param("local_trace_dir", str(self.trace_dir))
            self.mlflow.log_metric("total_tasks", self.total_tasks)
        except Exception as error:
            self.mlflow = None
            self.active_run = None
            print(f"MLflow logging disabled for this τ³-Bench run: {error}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.active_run is not None:
            return self.active_run.__exit__(exc_type, exc, tb)
        return False

    def _log_param(self, name: str, value: Any) -> None:
        if self.mlflow is None:
            return
        if isinstance(value, (dict, list)):
            rendered = json.dumps(make_json_safe(value), ensure_ascii=False)
        else:
            rendered = str(value)
        self.mlflow.log_param(name, rendered[:500])

    def _report_failure_once(self, error: Exception) -> None:
        if self.reported_failure:
            return
        self.reported_failure = True
        print(f"MLflow τ³-Bench logging failed; continuing eval without more MLflow logs: {error}")

    def log_task(
        self,
        row: dict[str, Any],
        *,
        step: int,
        completed_count: int,
        correct_count: int,
    ) -> None:
        if self.mlflow is None:
            return
        try:
            accuracy = correct_count / completed_count if completed_count else 0.0
            self.mlflow.log_metric("completed_count", completed_count, step=step)
            self.mlflow.log_metric("correct_count", correct_count, step=step)
            self.mlflow.log_metric("accuracy", accuracy, step=step)
            self.mlflow.log_metric("task_success", 1 if row.get("is_success") else 0, step=step)
            self.mlflow.log_metric("task_reward", float(row.get("reward") or 0.0), step=step)
            self.mlflow.log_metric(
                "task_duration_seconds",
                float(row.get("duration_seconds") or 0.0),
                step=step,
            )
            self.mlflow.log_metric(
                "task_message_count",
                int(row.get("message_count") or 0),
                step=step,
            )
            task_id = _safe_artifact_name(str(row.get("task_id", f"task_{step}")))
            artifact_payload = self._task_artifact_payload(row)
            self.mlflow.log_dict(artifact_payload, f"tasks/{task_id}.json")
        except Exception as error:
            self._report_failure_once(error)
            self.mlflow = None

    def _task_artifact_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.config is not None and self.config.log_full_artifacts:
            return make_json_safe(row)
        return {
            "task_id": row.get("task_id"),
            "reward": row.get("reward"),
            "is_success": row.get("is_success"),
            "termination_reason": row.get("termination_reason"),
            "duration_seconds": row.get("duration_seconds"),
            "message_count": row.get("message_count"),
            "reward_info": row.get("reward_info"),
            "error": row.get("error"),
        }

    def log_summary(self, summary: dict[str, Any]) -> None:
        if self.mlflow is None:
            return
        try:
            self.mlflow.log_metric("final_accuracy", float(summary.get("accuracy", 0.0)))
            self.mlflow.log_metric("final_correct_count", int(summary.get("correct_count", 0)))
            self.mlflow.log_metric("final_completed_count", int(summary.get("completed_count", 0)))
            self.mlflow.log_dict(
                {
                    key: value
                    for key, value in make_json_safe(summary).items()
                    if key != "task_results"
                },
                "summary.json",
            )
            if self.output_path.exists():
                self.mlflow.log_artifact(str(self.output_path), artifact_path="outputs")
            if (
                self.config is not None
                and self.config.log_full_artifacts
                and self.trace_dir is not None
                and self.trace_dir.exists()
            ):
                self.mlflow.log_artifacts(str(self.trace_dir), artifact_path="local_traces")
        except Exception as error:
            self._report_failure_once(error)
            self.mlflow = None
