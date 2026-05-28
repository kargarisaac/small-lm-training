from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import ast
import json
import subprocess
import sys

from .config import TAU_BENCH_REPO_REVISION, _configure_litellm_for_notebooks, _quiet_external_logs
from .qwen_format import ParsedQwenToolCall


__all__ = [
    "TauBenchRetailRuntime",
    "TauBenchRetailDomain",
    "load_tau_bench_retail_runtime",
    "load_tau_bench_retail_domain",
    "retail_agent_system_prompt",
    "coerce_tau_bench_retail_call_arguments",
]


@dataclass(frozen=True)
class TauBenchRetailRuntime:
    source_checkout: Path
    get_environment: Any
    get_tasks: Any
    UserSimulator: Any
    Orchestrator: Any
    EvaluationType: Any
    run_simulation: Any
    DEFAULT_FIRST_AGENT_MESSAGE: Any
    AssistantMessage: Any
    Message: Any
    MultiToolMessage: Any
    SystemMessage: Any
    ToolCall: Any
    ToolMessage: Any
    UserMessage: Any
    is_valid_agent_history_message: Any


@dataclass(frozen=True)
class TauBenchRetailDomain:
    runtime: TauBenchRetailRuntime
    data_dir: Path
    source_dir: Path
    paths: dict[str, Path]
    environment: Any
    policy: str
    tools: list[dict[str, Any]]
    tool_schema_by_name: dict[str, dict[str, Any]]


def _ensure_pinned_tau2_source_checkout(
    data_dir: Path,
    revision: str,
    repo_url: str = "https://github.com/sierra-research/tau2-bench.git",
) -> Path:
    checkout = data_dir / "external" / "tau2-bench"

    def checkout_matches_revision(candidate: Path) -> bool:
        try:
            result = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            return False
        return result.stdout.strip() == revision

    checkout.parent.mkdir(parents=True, exist_ok=True)
    if not (checkout / ".git").exists():
        subprocess.run(["git", "clone", repo_url, str(checkout)], check=True)

    if not checkout_matches_revision(checkout):
        subprocess.run(["git", "-C", str(checkout), "fetch", "origin", revision], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", revision], check=True)

    source_path = str(checkout / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    return checkout


def load_tau_bench_retail_runtime(data_dir: Path, revision: str) -> TauBenchRetailRuntime:
    _configure_litellm_for_notebooks()
    source_checkout = _ensure_pinned_tau2_source_checkout(data_dir, revision)

    with _quiet_external_logs():
        from tau2.data_model.message import (  # noqa: PLC0415
            AssistantMessage,
            Message,
            MultiToolMessage,
            SystemMessage,
            ToolCall,
            ToolMessage,
            UserMessage,
        )
        from tau2.agent.base_agent import is_valid_agent_history_message  # noqa: PLC0415
        from tau2.domains.retail.environment import get_environment, get_tasks  # noqa: PLC0415
        from tau2.evaluator.evaluator import EvaluationType  # noqa: PLC0415
        from tau2.orchestrator.orchestrator import (  # noqa: PLC0415
            DEFAULT_FIRST_AGENT_MESSAGE,
            Orchestrator,
        )
        from tau2.runner.simulation import run_simulation  # noqa: PLC0415
        from tau2.user.user_simulator import UserSimulator  # noqa: PLC0415

    return TauBenchRetailRuntime(
        source_checkout=source_checkout,
        get_environment=get_environment,
        get_tasks=get_tasks,
        UserSimulator=UserSimulator,
        Orchestrator=Orchestrator,
        EvaluationType=EvaluationType,
        run_simulation=run_simulation,
        DEFAULT_FIRST_AGENT_MESSAGE=DEFAULT_FIRST_AGENT_MESSAGE,
        AssistantMessage=AssistantMessage,
        Message=Message,
        MultiToolMessage=MultiToolMessage,
        SystemMessage=SystemMessage,
        ToolCall=ToolCall,
        ToolMessage=ToolMessage,
        UserMessage=UserMessage,
        is_valid_agent_history_message=is_valid_agent_history_message,
    )


def _tau_tool_to_qwen_schema(tool: Any) -> dict[str, Any]:
    schema = tool.openai_schema
    function_schema = schema.get("function", schema)
    return {
        "name": function_schema["name"],
        "description": function_schema.get("description", ""),
        "parameters": function_schema.get("parameters", {"type": "object", "properties": {}, "required": []}),
    }


def load_tau_bench_retail_domain(
    data_dir: Path,
    revision: str = TAU_BENCH_REPO_REVISION,
) -> TauBenchRetailDomain:
    runtime = load_tau_bench_retail_runtime(data_dir, revision)
    retail_data_dir = runtime.source_checkout / "data" / "tau2" / "domains" / "retail"
    retail_source_dir = runtime.source_checkout / "src" / "tau2" / "domains" / "retail"
    paths = {
        "tasks": retail_data_dir / "tasks.json",
        "splits": retail_data_dir / "split_tasks.json",
        "policy": retail_data_dir / "policy.md",
        "db": retail_data_dir / "db.json",
    }
    with _quiet_external_logs():
        environment = runtime.get_environment()
    policy = environment.get_policy()
    tools = [_tau_tool_to_qwen_schema(tool) for tool in environment.get_tools()]
    return TauBenchRetailDomain(
        runtime=runtime,
        data_dir=retail_data_dir,
        source_dir=retail_source_dir,
        paths=paths,
        environment=environment,
        policy=policy,
        tools=tools,
        tool_schema_by_name={tool["name"]: tool for tool in tools},
    )


def retail_agent_system_prompt(domain_policy: str) -> str:
    return (
        "You are a retail customer-support action policy agent.\n"
        "Follow the retail policy exactly. Use tools when tool results are needed.\n"
        "If you call a tool, output only the tool call. If you need user confirmation "
        "or the task is complete, answer the user in natural language.\n\n"
        "# Retail policy\n"
        f"{domain_policy}"
    )


def _coerce_tau_bench_retail_argument_value(value: str, schema: dict[str, Any]) -> Any:
    target_type = schema.get("type", "string")
    if target_type == "integer":
        return int(value)
    if target_type == "number":
        return float(value)
    if target_type == "boolean":
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise ValueError(f"Cannot coerce {value!r} to boolean.")
    if target_type in {"array", "object", "dict"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)
    return value


def coerce_tau_bench_retail_call_arguments(
    call: ParsedQwenToolCall,
    tool_schema_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if call.name not in tool_schema_by_name:
        raise ValueError(f"Unknown retail tool: {call.name}")

    tool_schema = tool_schema_by_name[call.name]
    parameter_schemas = tool_schema.get("parameters", {}).get("properties", {})
    required = set(tool_schema.get("parameters", {}).get("required", []))
    provided = set(call.arguments)
    missing = sorted(required - provided)
    if missing:
        raise ValueError(f"Missing required parameters for {call.name}: {missing}")

    coerced: dict[str, Any] = {}
    for name, value in call.arguments.items():
        if name not in parameter_schemas:
            raise ValueError(f"Unknown parameter for {call.name}: {name}")
        coerced[name] = _coerce_tau_bench_retail_argument_value(value, parameter_schemas[name])
    return coerced
