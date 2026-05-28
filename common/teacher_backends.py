from __future__ import annotations

from typing import Any, Callable
import json
import time
import urllib.error
import urllib.request

from .config import _GENERATION_STOP_STRINGS, TeacherConfig, _teacher_config_to_dict, make_json_safe
from .qwen_format import qwen_text_from_tool_call_parts


__all__ = [
    "teacher_runtime_is_configured",
    "teacher_server_health",
    "make_teacher_action_generator",
]


def teacher_runtime_is_configured(config: TeacherConfig) -> bool:
    if config.provider not in {"mlx_server", "mlx_raw_server", "ollama_raw", "vllm_raw", "chatgpt_raw"}:
        print(f"Unsupported teacher provider: {config.provider!r}")
        return False

    if config.provider == "chatgpt_raw":
        try:
            with urllib.request.urlopen(f"{_chatgpt_api_base(config)}/models", timeout=10) as response:
                if response.status != 200:
                    print(f"ChatGPT shim health check returned HTTP {response.status}.")
                    return False
                json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            print(f"Could not reach the ChatGPT subscription shim at {config.server_base_url}.")
            print("Start the shim from the notebook, then rerun this cell.")
            print("Error:", error)
            return False
        except json.JSONDecodeError as error:
            print(f"ChatGPT shim models endpoint returned invalid JSON: {error}")
            return False
        return True

    if config.provider == "vllm_raw":
        try:
            with urllib.request.urlopen(f"{config.server_base_url}/v1/models", timeout=10) as response:
                if response.status != 200:
                    print(f"vLLM health check returned HTTP {response.status}.")
                    return False
                models_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            print(f"Could not reach vLLM at {config.server_base_url}.")
            print("Start vLLM, then rerun the cell or script.")
            print("source /Users/kargarisaac/.venv-vllm-metal/bin/activate")
            print("Error:", error)
            return False
        except json.JSONDecodeError as error:
            print(f"vLLM models endpoint returned invalid JSON: {error}")
            return False

        requested_model = _vllm_request_model_name(config)
        available_models = {
            model.get("id")
            for model in models_payload.get("data", [])
            if isinstance(model, dict)
        }
        if requested_model not in available_models:
            print(f"vLLM is running, but model {requested_model!r} is not served.")
            print("Available models:", sorted(model for model in available_models if model))
            return False
        return True

    if config.provider == "ollama_raw":
        try:
            with urllib.request.urlopen(f"{config.server_base_url}/api/tags", timeout=10) as response:
                if response.status != 200:
                    print(f"Ollama health check returned HTTP {response.status}.")
                    return False
                tags_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            print(f"Could not reach Ollama at {config.server_base_url}.")
            print("Start Ollama, then rerun the cell or script.")
            print("Error:", error)
            return False
        except json.JSONDecodeError as error:
            print(f"Ollama tags endpoint returned invalid JSON: {error}")
            return False

        requested_model = _ollama_request_model_name(config)
        available_models = {
            model.get("name") or model.get("model")
            for model in tags_payload.get("models", [])
            if isinstance(model, dict)
        }
        if requested_model not in available_models:
            print(f"Ollama is running, but model {requested_model!r} is not installed.")
            print("Available models:", sorted(model for model in available_models if model))
            return False
        return True

    try:
        with urllib.request.urlopen(f"{config.server_base_url}/health", timeout=10) as response:
            if response.status != 200:
                print(f"MLX teacher server health check returned HTTP {response.status}.")
                return False
            health_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        print(f"Could not reach the MLX teacher server at {config.server_base_url}.")
        print("Start it in another terminal with:")
        if config.provider == "mlx_raw_server":
            print("uv run python scripts/serve_teacher_mlx_raw.py")
        else:
            print("uv run python scripts/serve_teacher_mlx.py")
        print("Error:", error)
        return False
    except json.JSONDecodeError as error:
        print(f"MLX teacher server health check returned invalid JSON: {error}")
        return False

    if config.provider == "mlx_raw_server" and "model" not in health_payload:
        print(f"The server at {config.server_base_url} is alive, but it is not the raw MLX teacher server.")
        print("Start the raw server with:")
        print("uv run python scripts/serve_teacher_mlx_raw.py")
        print("Raw server health should include a model field.")
        return False

    return True


def teacher_server_health(config: TeacherConfig) -> dict[str, Any] | None:
    try:
        if config.provider == "chatgpt_raw":
            path = "/models"
            base_url = _chatgpt_api_base(config)
        elif config.provider == "ollama_raw":
            path = "/api/tags"
            base_url = config.server_base_url
        elif config.provider == "vllm_raw":
            path = "/v1/models"
            base_url = config.server_base_url
        else:
            path = "/health"
            base_url = config.server_base_url
        with urllib.request.urlopen(f"{base_url}{path}", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if config.provider == "chatgpt_raw":
            models = [
                model.get("id")
                for model in payload.get("data", [])
                if isinstance(model, dict) and model.get("id")
            ]
            return {
                "model": config.model_name,
                "models": models,
                "raw": payload,
            }
        if config.provider == "vllm_raw":
            models = [
                model.get("id")
                for model in payload.get("data", [])
                if isinstance(model, dict) and model.get("id")
            ]
            return {
                "model": models[0] if models else config.model_name,
                "models": models,
                "raw": payload,
            }
        return payload
    except Exception:
        return None


def _build_teacher_chat_payload(
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    config: TeacherConfig,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    return {
        "model": config.request_model,
        "messages": messages,
        "tools": tools,
        "max_tokens": max_tokens or config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "stop": _GENERATION_STOP_STRINGS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
    }


def _request_mlx_chat_completion(payload: dict[str, Any], config: TeacherConfig) -> dict[str, Any]:
    request = urllib.request.Request(
        url=f"{config.server_base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MLX teacher HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach the MLX teacher server at {config.server_base_url}: {error}") from error


def _build_teacher_raw_completion_payload(
    prompt: str,
    config: TeacherConfig,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    return {
        "model": config.request_model,
        "prompt": prompt,
        "max_tokens": max_tokens or config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "stop": _GENERATION_STOP_STRINGS,
    }


def _ollama_request_model_name(config: TeacherConfig) -> str:
    if config.request_model and config.request_model != "default_model":
        return config.request_model
    return config.model_name


def _vllm_request_model_name(config: TeacherConfig) -> str:
    if config.request_model and config.request_model != "default_model":
        return config.request_model
    return config.model_name


def _chatgpt_api_base(config: TeacherConfig) -> str:
    base_url = config.server_base_url.rstrip("/")
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


def _build_ollama_raw_completion_payload(
    prompt: str,
    config: TeacherConfig,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "num_predict": max_tokens or config.max_new_tokens,
        "stop": _GENERATION_STOP_STRINGS,
    }
    return {
        "model": _ollama_request_model_name(config),
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": options,
    }


def _build_vllm_raw_completion_payload(
    prompt: str,
    config: TeacherConfig,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": _vllm_request_model_name(config),
        "prompt": prompt,
        "max_tokens": max_tokens or config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "stop": _GENERATION_STOP_STRINGS,
    }
    if config.top_k >= 0:
        payload["top_k"] = config.top_k
    return payload


def _build_chatgpt_raw_completion_payload(
    prompt: str,
    config: TeacherConfig,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.request_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a hosted teacher baseline inside a tool-use benchmark harness. "
                    "Continue the rendered Qwen tool-use prompt. Return only the assistant "
                    "completion text. If an action is needed, use the Qwen <tool_call> XML "
                    "format already shown in the prompt."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens or config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "stream": False,
    }
    if config.reasoning_effort:
        payload["reasoning"] = {"effort": config.reasoning_effort}
    return payload


def _request_mlx_raw_completion(payload: dict[str, Any], config: TeacherConfig) -> dict[str, Any]:
    request = urllib.request.Request(
        url=f"{config.server_base_url}/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            detail = error.read().decode("utf-8", errors="replace")
        except Exception as read_error:
            detail = f"<could not read error body: {read_error}>"
        raise RuntimeError(f"MLX teacher HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach the MLX teacher server at {config.server_base_url}: {error}") from error


def _request_ollama_raw_completion(payload: dict[str, Any], config: TeacherConfig) -> dict[str, Any]:
    request = urllib.request.Request(
        url=f"{config.server_base_url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            detail = error.read().decode("utf-8", errors="replace")
        except Exception as read_error:
            detail = f"<could not read error body: {read_error}>"
        raise RuntimeError(f"Ollama teacher HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach Ollama at {config.server_base_url}: {error}") from error


def _request_vllm_raw_completion(payload: dict[str, Any], config: TeacherConfig) -> dict[str, Any]:
    request = urllib.request.Request(
        url=f"{config.server_base_url}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            detail = error.read().decode("utf-8", errors="replace")
        except Exception as read_error:
            detail = f"<could not read error body: {read_error}>"
        raise RuntimeError(f"vLLM teacher HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach vLLM at {config.server_base_url}: {error}") from error


def _request_chatgpt_raw_completion(payload: dict[str, Any], config: TeacherConfig) -> dict[str, Any]:
    endpoint_url = f"{_chatgpt_api_base(config)}/chat/completions"
    request = urllib.request.Request(
        url=endpoint_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            detail = error.read().decode("utf-8", errors="replace")
        except Exception as read_error:
            detail = f"<could not read error body: {read_error}>"
        raise RuntimeError(f"ChatGPT teacher HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach ChatGPT shim at {config.server_base_url}: {error}") from error


def _record_endpoint_event(
    endpoint_event_sink: Callable[[dict[str, Any]], None] | None,
    event: dict[str, Any],
) -> None:
    if endpoint_event_sink is not None:
        endpoint_event_sink(make_json_safe(event))


def _qwen_text_from_mlx_raw_completion(response_payload: dict[str, Any]) -> str:
    text = response_payload.get("text")
    if text is None:
        raise RuntimeError(f"MLX raw teacher response has no text field: {response_payload}")
    return str(text)


def _qwen_text_from_ollama_raw_completion(response_payload: dict[str, Any]) -> str:
    text = response_payload.get("response")
    if text is None:
        raise RuntimeError(f"Ollama raw teacher response has no response field: {response_payload}")
    return str(text)


def _qwen_text_from_vllm_raw_completion(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices", [])
    if not choices:
        raise RuntimeError(f"vLLM raw teacher response has no choices: {response_payload}")

    text = choices[0].get("text")
    if text is None:
        raise RuntimeError(f"vLLM raw teacher choice has no text field: {choices[0]}")
    return str(text)


def _qwen_text_from_openai_chat_completion(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices", [])
    if not choices:
        raise RuntimeError(f"OpenAI-compatible chat response has no choices: {response_payload}")

    message = choices[0].get("message", {})
    content = message.get("content")
    if content is None:
        raise RuntimeError(f"OpenAI-compatible chat response has no message content: {choices[0]}")
    return str(content)


def _qwen_text_from_mlx_chat_response(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices", [])
    if not choices:
        raise RuntimeError(f"MLX teacher response has no choices: {response_payload}")

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        rendered_tool_calls = []
        for tool_call in tool_calls:
            function_payload = tool_call.get("function", {})
            function_name = function_payload.get("name")
            if not function_name:
                raise RuntimeError(f"MLX tool call has no function name: {tool_call}")

            raw_arguments = function_payload.get("arguments", {})
            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                raise RuntimeError(f"MLX tool call arguments are not a dict or JSON string: {tool_call}")
            if not isinstance(arguments, dict):
                raise RuntimeError(f"MLX tool call arguments did not decode to an object: {tool_call}")

            rendered_tool_calls.append(qwen_text_from_tool_call_parts(function_name, arguments))
        return "\n".join(rendered_tool_calls)

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning

    if content is not None:
        return str(content)

    if choices[0].get("finish_reason") == "tool_calls":
        return ""

    raise RuntimeError(f"MLX teacher message has neither content, reasoning, nor tool_calls: {choices[0]}")


def _mlx_server_chat_completion(
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    config: TeacherConfig,
    seed: int | None = None,
    endpoint_event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    payload = _build_teacher_chat_payload(messages=messages, tools=tools, config=config)
    if seed is not None:
        payload["seed"] = seed
    endpoint_url = f"{config.server_base_url}/v1/chat/completions"
    started_at = time.time()
    try:
        response_payload = _request_mlx_chat_completion(payload, config=config)
    except Exception as error:
        _record_endpoint_event(
            endpoint_event_sink,
            {
                "provider": config.provider,
                "endpoint_url": endpoint_url,
                "status": "error",
                "elapsed_seconds": round(time.time() - started_at, 3),
                "request_payload": payload,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise
    _record_endpoint_event(
        endpoint_event_sink,
        {
            "provider": config.provider,
            "endpoint_url": endpoint_url,
            "status": "ok",
            "elapsed_seconds": round(time.time() - started_at, 3),
            "request_payload": payload,
            "response_payload": response_payload,
        },
    )
    return _qwen_text_from_mlx_chat_response(response_payload)


def _mlx_raw_server_completion(
    prompt: str | None,
    config: TeacherConfig,
    seed: int | None = None,
    endpoint_event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    if prompt is None:
        raise RuntimeError("The raw MLX teacher provider requires the rendered Qwen prompt.")
    payload = _build_teacher_raw_completion_payload(prompt=prompt, config=config)
    if seed is not None:
        payload["seed"] = seed
    endpoint_url = f"{config.server_base_url}/generate"
    started_at = time.time()
    try:
        response_payload = _request_mlx_raw_completion(payload, config=config)
    except Exception as error:
        _record_endpoint_event(
            endpoint_event_sink,
            {
                "provider": config.provider,
                "endpoint_url": endpoint_url,
                "status": "error",
                "elapsed_seconds": round(time.time() - started_at, 3),
                "request_payload": payload,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise
    _record_endpoint_event(
        endpoint_event_sink,
        {
            "provider": config.provider,
            "endpoint_url": endpoint_url,
            "status": "ok",
            "elapsed_seconds": round(time.time() - started_at, 3),
            "request_payload": payload,
            "response_payload": response_payload,
        },
    )
    return _qwen_text_from_mlx_raw_completion(response_payload)


def _ollama_raw_completion(
    prompt: str | None,
    config: TeacherConfig,
    seed: int | None = None,
    endpoint_event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    if prompt is None:
        raise RuntimeError("The raw Ollama teacher provider requires the rendered Qwen prompt.")
    payload = _build_ollama_raw_completion_payload(prompt=prompt, config=config)
    if seed is not None:
        payload["options"]["seed"] = seed
    endpoint_url = f"{config.server_base_url}/api/generate"
    started_at = time.time()
    try:
        response_payload = _request_ollama_raw_completion(payload, config=config)
    except Exception as error:
        _record_endpoint_event(
            endpoint_event_sink,
            {
                "provider": config.provider,
                "endpoint_url": endpoint_url,
                "status": "error",
                "elapsed_seconds": round(time.time() - started_at, 3),
                "request_payload": payload,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise
    _record_endpoint_event(
        endpoint_event_sink,
        {
            "provider": config.provider,
            "endpoint_url": endpoint_url,
            "status": "ok",
            "elapsed_seconds": round(time.time() - started_at, 3),
            "request_payload": payload,
            "response_payload": response_payload,
        },
    )
    return _qwen_text_from_ollama_raw_completion(response_payload)


def _vllm_raw_completion(
    prompt: str | None,
    config: TeacherConfig,
    seed: int | None = None,
    endpoint_event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    if prompt is None:
        raise RuntimeError("The raw vLLM teacher provider requires the rendered Qwen prompt.")
    payload = _build_vllm_raw_completion_payload(prompt=prompt, config=config)
    if seed is not None:
        payload["seed"] = seed
    endpoint_url = f"{config.server_base_url}/v1/completions"
    started_at = time.time()
    try:
        response_payload = _request_vllm_raw_completion(payload, config=config)
    except Exception as error:
        _record_endpoint_event(
            endpoint_event_sink,
            {
                "provider": config.provider,
                "endpoint_url": endpoint_url,
                "status": "error",
                "elapsed_seconds": round(time.time() - started_at, 3),
                "request_payload": payload,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise
    _record_endpoint_event(
        endpoint_event_sink,
        {
            "provider": config.provider,
            "endpoint_url": endpoint_url,
            "status": "ok",
            "elapsed_seconds": round(time.time() - started_at, 3),
            "request_payload": payload,
            "response_payload": response_payload,
        },
    )
    return _qwen_text_from_vllm_raw_completion(response_payload)


def _chatgpt_raw_completion(
    prompt: str | None,
    config: TeacherConfig,
    endpoint_event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    if prompt is None:
        raise RuntimeError("The ChatGPT teacher provider requires the rendered Qwen prompt.")
    payload = _build_chatgpt_raw_completion_payload(prompt=prompt, config=config)
    endpoint_url = f"{_chatgpt_api_base(config)}/chat/completions"
    started_at = time.time()
    try:
        response_payload = _request_chatgpt_raw_completion(payload, config=config)
    except Exception as error:
        _record_endpoint_event(
            endpoint_event_sink,
            {
                "provider": config.provider,
                "endpoint_url": endpoint_url,
                "status": "error",
                "elapsed_seconds": round(time.time() - started_at, 3),
                "request_payload": payload,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise
    _record_endpoint_event(
        endpoint_event_sink,
        {
            "provider": config.provider,
            "endpoint_url": endpoint_url,
            "status": "ok",
            "elapsed_seconds": round(time.time() - started_at, 3),
            "request_payload": payload,
            "response_payload": response_payload,
        },
    )
    return _qwen_text_from_openai_chat_completion(response_payload)


def make_teacher_action_generator(config: TeacherConfig) -> Callable[..., str]:
    def generate_teacher_action(
        *,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        prompt: str | None = None,
    ) -> str:
        generate_teacher_action.last_endpoint_call = None

        def record_call(event: dict[str, Any]) -> None:
            generate_teacher_action.last_endpoint_call = event

        if config.provider == "mlx_raw_server":
            return _mlx_raw_server_completion(
                prompt=prompt,
                config=config,
                endpoint_event_sink=record_call,
            )
        if config.provider == "ollama_raw":
            return _ollama_raw_completion(
                prompt=prompt,
                config=config,
                endpoint_event_sink=record_call,
            )
        if config.provider == "vllm_raw":
            return _vllm_raw_completion(
                prompt=prompt,
                config=config,
                endpoint_event_sink=record_call,
            )
        if config.provider == "chatgpt_raw":
            return _chatgpt_raw_completion(
                prompt=prompt,
                config=config,
                endpoint_event_sink=record_call,
            )
        if config.provider == "mlx_server":
            return _mlx_server_chat_completion(
                messages=messages,
                tools=tools,
                config=config,
                endpoint_event_sink=record_call,
            )
        raise RuntimeError(f"Unsupported teacher provider: {config.provider!r}")

    generate_teacher_action.generation_config = _teacher_config_to_dict(config)
    generate_teacher_action.last_endpoint_call = None
    return generate_teacher_action
