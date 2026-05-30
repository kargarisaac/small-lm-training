from __future__ import annotations

from typing import Any, Callable
import json
import os
import urllib.request

from . import nestful


Generator = Callable[[dict[str, Any]], str]


def make_generator(
    *,
    backend: str,
    model_name: str,
    adapter: str | None = None,
    max_new_tokens: int = 128,
    base_url: str | None = None,
    api_key_env: str | None = None,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
) -> Generator:
    if backend == "hf":
        return hf_generator(model_name, adapter, max_new_tokens)
    if backend == "mlx":
        return mlx_generator(model_name, adapter, max_new_tokens)
    if backend == "openai":
        if not base_url:
            raise ValueError("--base-url is required for --backend openai.")
        return openai_compatible_generator(model_name, base_url, api_key_env, max_new_tokens, temperature, reasoning_effort)
    raise ValueError(f"Unsupported backend: {backend}")


def hf_generator(model_name: str, adapter: str | None, max_new_tokens: int) -> Generator:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, trust_remote_code=True)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()

    def generate(row: dict[str, Any]) -> str:
        prompt = nestful.render_prompt(tokenizer, row)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_ids, skip_special_tokens=False)

    return generate


def mlx_generator(model_name: str, adapter: str | None, max_new_tokens: int) -> Generator:
    from mlx_lm import generate as mlx_lm_generate
    from mlx_lm import load as mlx_lm_load

    model, tokenizer = mlx_lm_load(model_name, adapter_path=adapter)

    def generate(row: dict[str, Any]) -> str:
        prompt = nestful.render_prompt(tokenizer, row)
        return mlx_lm_generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=False)

    return generate


def openai_compatible_generator(
    model_name: str,
    base_url: str,
    api_key_env: str | None,
    max_new_tokens: int,
    temperature: float,
    reasoning_effort: str | None,
) -> Generator:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    api_key = os.environ.get(api_key_env) if api_key_env else None

    def generate(row: dict[str, Any]) -> str:
        payload = {
            "model": model_name,
            "messages": row["messages"][:-1],
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
        message = body["choices"][0]["message"]
        return message.get("content") or ""

    return generate
