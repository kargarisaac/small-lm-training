from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any
import hashlib
import json
import time
import types

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from mlx_lm.tuner.datasets import CacheDataset, load_dataset
from mlx_lm.tuner.trainer import default_loss, grad_checkpoint
from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers, print_trainable_parameters
from mlx_lm.utils import load, save_config


@dataclass
class ResumableLoraConfig:
    model: str
    data_dir: Path
    adapter_path: Path
    total_iters: int
    max_seq_length: int
    batch_size: int = 1
    grad_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    optimizer: str = "adam"
    optimizer_config: dict[str, Any] = field(default_factory=dict)
    lr_schedule: Any | None = None
    val_batches: int = 1
    steps_per_report: int = 1
    steps_per_eval: int = 25
    save_every: int = 25
    num_layers: int = 1
    lora_rank: int = 8
    lora_scale: float = 20.0
    lora_dropout: float = 0.0
    fine_tune_type: str = "lora"
    mask_prompt: bool = True
    seed: int = 0
    grad_checkpoint: bool = True
    clear_cache_threshold: int = 0
    resume: str | None = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256_path(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_fingerprint(data_dir: Path) -> dict[str, str | None]:
    return {
        name: _sha256_path(data_dir / f"{name}.jsonl")
        for name in ("train", "valid", "test")
    }


def _latest_checkpoint_path(config: ResumableLoraConfig) -> Path | None:
    checkpoint_root = config.adapter_path / "training_checkpoints"
    latest_path = checkpoint_root / "latest_checkpoint.json"
    if latest_path.exists():
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        checkpoint_path = payload.get("checkpoint_path")
        candidate = Path(checkpoint_path) if checkpoint_path else None
        if candidate and candidate.exists():
            return candidate

    candidates = sorted(checkpoint_root.glob("checkpoint_*"))
    return candidates[-1] if candidates else None


def _resolve_resume_checkpoint(config: ResumableLoraConfig) -> Path | None:
    resume = (config.resume or "none").strip()
    if resume in {"", "none", "false", "0"}:
        return None
    if resume == "latest":
        checkpoint = _latest_checkpoint_path(config)
        if checkpoint is None:
            raise FileNotFoundError(f"No latest checkpoint found in {config.adapter_path / 'training_checkpoints'}")
        return checkpoint

    checkpoint = Path(resume).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _adapter_config(config: ResumableLoraConfig) -> dict[str, Any]:
    return {
        "fine_tune_type": config.fine_tune_type,
        "num_layers": config.num_layers,
        "lora_parameters": {
            "rank": config.lora_rank,
            "scale": config.lora_scale,
            "dropout": config.lora_dropout,
        },
        "model": config.model,
        "data": str(config.data_dir),
        "mask_prompt": config.mask_prompt,
        "batch_size": config.batch_size,
        "grad_accumulation_steps": config.grad_accumulation_steps,
        "max_seq_length": config.max_seq_length,
        "learning_rate": config.learning_rate,
    }


def _training_args_namespace(config: ResumableLoraConfig) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        train=True,
        test=False,
        data=str(config.data_dir),
        mask_prompt=config.mask_prompt,
        prompt_feature="prompt",
        text_feature="text",
        completion_feature="completion",
        chat_feature="messages",
        hf_dataset=False,
    )


def _build_optimizer(config: ResumableLoraConfig):
    learning_rate = build_schedule(config.lr_schedule) if config.lr_schedule else config.learning_rate
    optimizer_config = config.optimizer_config.get(config.optimizer, config.optimizer_config)
    optimizer_name = config.optimizer.lower()
    if optimizer_name == "adam":
        optimizer_class = optim.Adam
    elif optimizer_name == "adamw":
        optimizer_class = optim.AdamW
    elif optimizer_name == "muon":
        optimizer_class = optim.Muon
    elif optimizer_name == "sgd":
        optimizer_class = optim.SGD
    elif optimizer_name == "adafactor":
        optimizer_class = optim.Adafactor
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    return optimizer_class(learning_rate=learning_rate, **optimizer_config)


def _prepare_model(config: ResumableLoraConfig):
    print("Loading pretrained model")
    model, tokenizer = load(config.model, tokenizer_config={"trust_remote_code": True})
    mx.random.seed(config.seed)
    np.random.seed(config.seed)

    model.freeze()
    if config.num_layers > len(model.layers):
        raise ValueError(
            f"Requested to train {config.num_layers} layers but the model only has {len(model.layers)} layers."
        )

    if config.fine_tune_type == "full":
        for layer in model.layers[-max(config.num_layers, 0) :]:
            layer.unfreeze()
    elif config.fine_tune_type == "lora":
        linear_to_lora_layers(model, config.num_layers, _adapter_config(config)["lora_parameters"])
    else:
        raise ValueError(f"Unsupported fine_tune_type: {config.fine_tune_type}")

    print_trainable_parameters(model)
    return model, tokenizer


def _batch_plan(dataset: CacheDataset, batch_size: int) -> list[list[int]]:
    lengths = [len(dataset[index][0]) for index in range(len(dataset))]
    sorted_indices = sorted(range(len(dataset)), key=lambda index: lengths[index])
    return [
        sorted_indices[start : start + batch_size]
        for start in range(0, len(sorted_indices) - batch_size + 1, batch_size)
    ]


def _batch_indices_for_step(plan: list[list[int]], *, step: int, seed: int) -> list[int]:
    if not plan:
        raise ValueError("Dataset must have at least one complete batch.")
    epoch = (step - 1) // len(plan)
    position = (step - 1) % len(plan)
    rng = np.random.default_rng(seed + epoch)
    permutation = rng.permutation(len(plan))
    return plan[int(permutation[position])]


def _make_batch(dataset: CacheDataset, indices: list[int], max_seq_length: int):
    batch_items = [dataset[index] for index in indices]
    tokens, offsets = zip(*batch_items)
    lengths = [len(item) for item in tokens]
    max_length_in_batch = 1 + 32 * ((max(lengths) + 31) // 32)
    max_length_in_batch = min(max_length_in_batch, max_seq_length)

    batch_array = np.zeros((len(tokens), max_length_in_batch), np.int32)
    truncated_lengths: list[int] = []
    for row_index, token_ids in enumerate(tokens):
        truncated_length = min(len(token_ids), max_seq_length)
        batch_array[row_index, :truncated_length] = token_ids[:truncated_length]
        truncated_lengths.append(truncated_length)

    return mx.array(batch_array), mx.array(list(zip(offsets, truncated_lengths)))


def _evaluate_loss(
    *,
    model,
    dataset: CacheDataset,
    plan: list[list[int]],
    batch_size: int,
    num_batches: int,
    max_seq_length: int,
    clear_cache_threshold: int,
) -> float:
    if not dataset:
        raise ValueError("Validation dataset is empty.")

    model.eval()
    total_loss = mx.array(0.0)
    total_tokens = mx.array(0)
    available_batches = len(plan)
    batches_to_run = available_batches if num_batches == -1 else min(available_batches, num_batches)
    for batch_number in range(1, batches_to_run + 1):
        batch = _make_batch(dataset, plan[batch_number - 1], max_seq_length)
        loss, tokens = default_loss(model, *batch)
        total_loss += loss * tokens
        total_tokens += tokens
        mx.eval(total_loss, total_tokens)
        if clear_cache_threshold >= 0 and mx.get_cache_memory() > clear_cache_threshold:
            mx.clear_cache()

    model.train()
    return (total_loss / total_tokens).item()


def _save_checkpoint(
    *,
    config: ResumableLoraConfig,
    model,
    optimizer,
    grad_accum: Any,
    trainer_state: dict[str, Any],
    dataset_fingerprint: dict[str, str | None],
) -> Path:
    checkpoint_root = config.adapter_path / "training_checkpoints"
    checkpoint_dir = checkpoint_root / f"checkpoint_{trainer_state['global_step']:07d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.adapter_path.mkdir(parents=True, exist_ok=True)

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(checkpoint_dir / "adapters.safetensors"), adapter_weights)
    mx.save_safetensors(str(config.adapter_path / "adapters.safetensors"), adapter_weights)
    mx.save_safetensors(
        str(checkpoint_dir / "optimizer.safetensors"),
        tree_flatten(optimizer.state, destination={}),
    )
    mx.save_safetensors(
        str(checkpoint_dir / "random_state.safetensors"),
        tree_flatten(mx.random.state, destination={}),
    )
    if grad_accum is not None:
        mx.save_safetensors(
            str(checkpoint_dir / "grad_accum.safetensors"),
            tree_flatten(grad_accum, destination={}),
        )

    state_payload = {
        **trainer_state,
        "has_grad_accum": grad_accum is not None,
        "dataset_fingerprint": dataset_fingerprint,
        "config": _json_safe(asdict(config)),
    }
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(_json_safe(state_payload), indent=2),
        encoding="utf-8",
    )
    save_config(_adapter_config(config), config.adapter_path / "adapter_config.json")

    latest_path = checkpoint_root / "latest_checkpoint.json"
    latest_path.write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_dir),
                "global_step": trainer_state["global_step"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return checkpoint_dir


def _load_checkpoint(
    *,
    checkpoint_dir: Path,
    config: ResumableLoraConfig,
    model,
    optimizer,
    dataset_fingerprint: dict[str, str | None],
) -> tuple[dict[str, Any], Any]:
    state_path = checkpoint_dir / "trainer_state.json"
    trainer_state = json.loads(state_path.read_text(encoding="utf-8"))
    if trainer_state.get("dataset_fingerprint") != dataset_fingerprint:
        raise RuntimeError(
            "Checkpoint dataset fingerprint does not match the current train/valid/test JSONL files."
        )

    print(f"Loading full training checkpoint from {checkpoint_dir}")
    model.load_weights(str(checkpoint_dir / "adapters.safetensors"), strict=False)
    optimizer.state = tree_unflatten(mx.load(str(checkpoint_dir / "optimizer.safetensors")))
    mx.random.state = tree_unflatten(mx.load(str(checkpoint_dir / "random_state.safetensors")))
    grad_accum_path = checkpoint_dir / "grad_accum.safetensors"
    grad_accum = (
        tree_unflatten(mx.load(str(grad_accum_path)))
        if trainer_state.get("has_grad_accum") and grad_accum_path.exists()
        else None
    )
    return trainer_state, grad_accum


def _append_history_line(history_path: Path, row: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def run_resumable_lora_training(config: ResumableLoraConfig) -> dict[str, Any]:
    if config.total_iters < 1:
        raise ValueError("total_iters must be at least 1.")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if config.grad_accumulation_steps < 1:
        raise ValueError("grad_accumulation_steps must be at least 1.")

    config.data_dir = Path(config.data_dir)
    config.adapter_path = Path(config.adapter_path)
    config.adapter_path.mkdir(parents=True, exist_ok=True)
    checkpoint_root = config.adapter_path / "training_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    save_config(_adapter_config(config), config.adapter_path / "adapter_config.json")

    dataset_fingerprint = _dataset_fingerprint(config.data_dir)
    model, tokenizer = _prepare_model(config)
    print("Loading datasets")
    train_set, valid_set, _test_set = load_dataset(_training_args_namespace(config), tokenizer)
    train_dataset = CacheDataset(train_set)
    valid_dataset = CacheDataset(valid_set) if valid_set else None
    train_plan = _batch_plan(train_dataset, config.batch_size)
    valid_plan = _batch_plan(valid_dataset, config.batch_size) if valid_dataset else []

    if not train_plan:
        raise ValueError(f"Dataset must have at least batch_size={config.batch_size} examples.")

    optimizer = _build_optimizer(config)
    resume_checkpoint = _resolve_resume_checkpoint(config)
    history_path = config.adapter_path / "training_history.jsonl"

    trainer_state: dict[str, Any] = {
        "global_step": 0,
        "trained_tokens": 0,
        "train_history": [],
        "val_history": [],
        "report_loss_sum": 0.0,
        "report_token_count": 0,
        "report_step_count": 0,
        "report_train_time": 0.0,
    }
    grad_accum = None
    if resume_checkpoint is None:
        history_path.write_text("", encoding="utf-8")
        (checkpoint_root / "latest_checkpoint.json").write_text(
            json.dumps(
                {
                    "checkpoint_path": None,
                    "global_step": 0,
                    "status": "fresh_run_started",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("Starting from scratch.")
    else:
        trainer_state, grad_accum = _load_checkpoint(
            checkpoint_dir=resume_checkpoint,
            config=config,
            model=model,
            optimizer=optimizer,
            dataset_fingerprint=dataset_fingerprint,
        )

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    if config.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    loss_value_and_grad = nn.value_and_grad(model, default_loss)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, previous_grad, do_update):
        (loss_value, tokens), grad = loss_value_and_grad(model, *batch)
        if previous_grad is not None:
            grad = tree_map(lambda current, previous: current + previous, grad, previous_grad)
        if do_update:
            if config.grad_accumulation_steps > 1:
                grad = tree_map(lambda value: value / config.grad_accumulation_steps, grad)
            optimizer.update(model, grad)
            grad = None
        return loss_value, tokens, grad

    model.train()
    start_step = int(trainer_state["global_step"])
    print(f"Starting training at iteration {start_step + 1}; target iteration {config.total_iters}.")

    for global_step in range(start_step + 1, config.total_iters + 1):
        if valid_dataset and (
            global_step == 1
            or global_step % config.steps_per_eval == 0
            or global_step == config.total_iters
        ):
            validation_started = time.perf_counter()
            val_loss = _evaluate_loss(
                model=model,
                dataset=valid_dataset,
                plan=valid_plan,
                batch_size=config.batch_size,
                num_batches=config.val_batches,
                max_seq_length=config.max_seq_length,
                clear_cache_threshold=config.clear_cache_threshold,
            )
            val_time = time.perf_counter() - validation_started
            val_row = {"iteration": global_step - 1, "val_loss": val_loss, "val_time": val_time}
            trainer_state["val_history"].append(val_row)
            _append_history_line(history_path, {"kind": "validation", **val_row})
            print(f"Iter {global_step}: Val loss {val_loss:.3f}, Val took {val_time:.3f}s", flush=True)

        batch_indices = _batch_indices_for_step(train_plan, step=global_step, seed=config.seed)
        batch = _make_batch(train_dataset, batch_indices, config.max_seq_length)

        train_started = time.perf_counter()
        loss_value, tokens, grad_accum = step(
            batch,
            grad_accum,
            global_step % config.grad_accumulation_steps == 0,
        )
        mx.eval(state, loss_value, tokens, grad_accum)
        if config.clear_cache_threshold >= 0 and mx.get_cache_memory() > config.clear_cache_threshold:
            mx.clear_cache()
        train_time = time.perf_counter() - train_started

        trainer_state["global_step"] = global_step
        trainer_state["report_loss_sum"] += loss_value.item()
        trainer_state["report_token_count"] += tokens.item()
        trainer_state["report_step_count"] += 1
        trainer_state["report_train_time"] += train_time

        if global_step % config.steps_per_report == 0 or global_step == config.total_iters:
            report_steps = int(trainer_state["report_step_count"])
            report_tokens = int(trainer_state["report_token_count"])
            report_time = float(trainer_state["report_train_time"])
            train_loss = float(trainer_state["report_loss_sum"]) / max(1, report_steps)
            trainer_state["trained_tokens"] += report_tokens
            row = {
                "iteration": global_step,
                "train_loss": train_loss,
                "learning_rate": optimizer.learning_rate.item(),
                "iterations_per_second": report_steps / report_time if report_time else 0.0,
                "tokens_per_second": report_tokens / report_time if report_time else 0.0,
                "trained_tokens": int(trainer_state["trained_tokens"]),
                "active_memory_gb": mx.get_active_memory() / 1e9,
                "cache_memory_gb": mx.get_cache_memory() / 1e9,
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
            }
            trainer_state["train_history"].append(row)
            _append_history_line(history_path, {"kind": "train", **row})
            print(
                f"Iter {global_step}: Train loss {train_loss:.3f}, "
                f"Learning Rate {row['learning_rate']:.3e}, "
                f"It/sec {row['iterations_per_second']:.3f}, "
                f"Tokens/sec {row['tokens_per_second']:.3f}, "
                f"Trained Tokens {row['trained_tokens']}, "
                f"Active mem {row['active_memory_gb']:.3f} GB, "
                f"Cache mem {row['cache_memory_gb']:.3f} GB, "
                f"Peak mem {row['peak_memory_gb']:.3f} GB",
                flush=True,
            )
            trainer_state["report_loss_sum"] = 0.0
            trainer_state["report_token_count"] = 0
            trainer_state["report_step_count"] = 0
            trainer_state["report_train_time"] = 0.0

        if global_step % config.save_every == 0 or global_step == config.total_iters:
            checkpoint_dir = _save_checkpoint(
                config=config,
                model=model,
                optimizer=optimizer,
                grad_accum=grad_accum,
                trainer_state=trainer_state,
                dataset_fingerprint=dataset_fingerprint,
            )
            print(f"Iter {global_step}: Saved full checkpoint to {checkpoint_dir}.")

    final_checkpoint = _save_checkpoint(
        config=config,
        model=model,
        optimizer=optimizer,
        grad_accum=grad_accum,
        trainer_state=trainer_state,
        dataset_fingerprint=dataset_fingerprint,
    )
    print(f"Saved final adapter to {config.adapter_path / 'adapters.safetensors'}.")
    return {
        "adapter_path": str(config.adapter_path),
        "checkpoint_root": str(checkpoint_root),
        "final_checkpoint": str(final_checkpoint),
        "history_path": str(history_path),
        "global_step": trainer_state["global_step"],
        "trained_tokens": trainer_state["trained_tokens"],
        "train_history": trainer_state["train_history"],
        "val_history": trainer_state["val_history"],
    }
