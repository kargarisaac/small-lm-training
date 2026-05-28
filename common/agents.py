from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4
import threading

from .config import MAX_NEW_TOKENS, TeacherConfig
from .qwen_format import parse_qwen_tool_calls, qwen_text_from_tool_call_parts, strip_generated_special_tokens
from .tau_runtime import TauBenchRetailRuntime, coerce_tau_bench_retail_call_arguments, retail_agent_system_prompt
from .teacher_backends import make_teacher_action_generator


@dataclass
class TauBenchRetailAgentState:
    system_messages: list[Any]
    messages: list[Any]


def _tau_message_to_qwen_message(message: Any, runtime: TauBenchRetailRuntime) -> dict[str, str] | None:
    if isinstance(message, runtime.SystemMessage):
        return {"role": "system", "content": message.content or ""}
    if isinstance(message, runtime.UserMessage):
        return {"role": "user", "content": message.content or ""}
    if isinstance(message, runtime.ToolMessage):
        return {"role": "tool", "content": message.content or ""}
    if isinstance(message, runtime.AssistantMessage):
        if message.tool_calls:
            content = "\n".join(
                qwen_text_from_tool_call_parts(tool_call.name, tool_call.arguments)
                for tool_call in message.tool_calls or []
            )
            return {"role": "assistant", "content": content}
        return {"role": "assistant", "content": message.content or ""}
    return None


class _TauBenchRetailQwenAgentBase:
    def __init__(
        self,
        *,
        runtime: TauBenchRetailRuntime,
        tokenizer: Any,
        qwen_tools: list[dict[str, Any]],
        tool_schema_by_name: dict[str, dict[str, Any]],
        domain_policy: str,
    ):
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.qwen_tools = qwen_tools
        self.tool_schema_by_name = tool_schema_by_name
        self.system_prompt = retail_agent_system_prompt(domain_policy)

    def get_init_state(self, message_history: list[Any] | None = None) -> TauBenchRetailAgentState:
        if message_history is None:
            message_history = []
        assert all(self.runtime.is_valid_agent_history_message(message) for message in message_history)
        return TauBenchRetailAgentState(
            system_messages=[self.runtime.SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
        )

    @classmethod
    def is_stop(cls, message: Any) -> bool:
        return bool(message.content and "###STOP###" in message.content)

    def set_seed(self, seed: int) -> None:
        return None

    def stop(self, message: Any, state: TauBenchRetailAgentState | None = None) -> None:
        return None

    def _prepare_messages(self, message: Any, state: TauBenchRetailAgentState) -> list[dict[str, str]]:
        if isinstance(message, self.runtime.MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is not None:
            state.messages.append(message)

        qwen_messages: list[dict[str, str]] = []
        for item in [*state.system_messages, *state.messages]:
            qwen_message = _tau_message_to_qwen_message(item, self.runtime)
            if qwen_message is not None:
                qwen_messages.append(qwen_message)
        return qwen_messages

    def _raw_completion(self, *, qwen_messages: list[dict[str, str]], prompt: str) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def generate_next_message(
        self,
        message: Any,
        state: TauBenchRetailAgentState,
    ) -> tuple[Any, TauBenchRetailAgentState]:
        qwen_messages = self._prepare_messages(message, state)
        prompt = self.tokenizer.apply_chat_template(
            qwen_messages,
            tools=self.qwen_tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        raw_output, extra_raw_data = self._raw_completion(qwen_messages=qwen_messages, prompt=prompt)
        completion_tokens = len(self.tokenizer.encode(raw_output, add_special_tokens=False))

        parsed = parse_qwen_tool_calls(raw_output)
        cleaned = strip_generated_special_tokens(raw_output)
        raw_data = {
            "raw_qwen_output": raw_output,
            "parse_errors": parsed.errors,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            **extra_raw_data,
        }

        tool_calls: list[Any] = []
        coercion_errors: list[str] = []
        for parsed_call in parsed.calls:
            try:
                arguments = coerce_tau_bench_retail_call_arguments(parsed_call, self.tool_schema_by_name)
                tool_calls.append(
                    self.runtime.ToolCall(
                        id=f"call_{uuid4().hex}",
                        name=parsed_call.name,
                        arguments=arguments,
                        requestor="assistant",
                    )
                )
            except Exception as exc:
                coercion_errors.append(f"{parsed_call.name}: {type(exc).__name__}: {exc}")

        if tool_calls and not coercion_errors:
            assistant_message = self.runtime.AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=tool_calls,
                raw_data=raw_data,
            )
        else:
            if coercion_errors:
                raw_data["coercion_errors"] = coercion_errors
            assistant_message = self.runtime.AssistantMessage(
                role="assistant",
                content=cleaned,
                raw_data=raw_data,
            )

        state.messages.append(assistant_message)
        return assistant_message, state


class TauBenchRetailQwenRawCompletionAgent(_TauBenchRetailQwenAgentBase):
    def __init__(
        self,
        *,
        runtime: TauBenchRetailRuntime,
        tokenizer: Any,
        qwen_tools: list[dict[str, Any]],
        tool_schema_by_name: dict[str, dict[str, Any]],
        domain_policy: str,
        teacher_config: TeacherConfig,
    ):
        super().__init__(
            runtime=runtime,
            tokenizer=tokenizer,
            qwen_tools=qwen_tools,
            tool_schema_by_name=tool_schema_by_name,
            domain_policy=domain_policy,
        )
        self.teacher_config = teacher_config
        self.generate_action = make_teacher_action_generator(teacher_config)
        self.seed: int | None = None

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    def _raw_completion(self, *, qwen_messages: list[dict[str, str]], prompt: str) -> tuple[str, dict[str, Any]]:
        raw_output = self.generate_action(messages=qwen_messages, tools=self.qwen_tools, prompt=prompt)
        return raw_output, {
            "generation_config": getattr(self.generate_action, "generation_config", {}),
            "endpoint_call": getattr(self.generate_action, "last_endpoint_call", None),
        }


class TauBenchRetailMlxStudentAgent(_TauBenchRetailQwenAgentBase):
    def __init__(
        self,
        *,
        runtime: TauBenchRetailRuntime,
        model: Any,
        tokenizer: Any,
        qwen_tools: list[dict[str, Any]],
        tool_schema_by_name: dict[str, dict[str, Any]],
        domain_policy: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        sampler: Any,
        generation_lock: threading.Lock | None = None,
    ):
        from mlx_lm import generate as mlx_generate  # noqa: PLC0415

        super().__init__(
            runtime=runtime,
            tokenizer=tokenizer,
            qwen_tools=qwen_tools,
            tool_schema_by_name=tool_schema_by_name,
            domain_policy=domain_policy,
        )
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.sampler = sampler
        self.generation_lock = generation_lock
        self.mlx_generate = mlx_generate

    def _raw_completion(self, *, qwen_messages: list[dict[str, str]], prompt: str) -> tuple[str, dict[str, Any]]:
        def generate() -> str:
            return self.mlx_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_new_tokens,
                sampler=self.sampler,
                verbose=False,
            )

        if self.generation_lock is None:
            return generate(), {}
        with self.generation_lock:
            return generate(), {}
