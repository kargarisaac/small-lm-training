from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import sft_rows


DEFAULT_NVIDIA_TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tau3 retail student with TRL/PEFT LoRA.")
    parser.add_argument("--student-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--teacher-model", default=DEFAULT_NVIDIA_TEACHER_MODEL)
    parser.add_argument("--teacher-provider", default="vllm_raw")
    parser.add_argument("--sft-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-seq-length", type=int, default=16500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=20)
    parser.add_argument("--lora-layers", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 means one pass over kept rows.")
    parser.add_argument("--resume", default=None, help="Use 'latest' or a checkpoint path.")
    return parser.parse_args()


def tokenize_row(row: dict, tokenizer, max_seq_length: int) -> dict | None:
    target_message = row["messages"][-1]
    if target_message["role"] != "assistant":
        raise ValueError("SFT target message must be an assistant message.")
    prompt_text = tokenizer.apply_chat_template(
        row["messages"][:-1],
        tools=row.get("tools"),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_message["content"], add_special_tokens=False)
    if tokenizer.eos_token_id is not None and (not target_ids or target_ids[-1] != tokenizer.eos_token_id):
        target_ids.append(tokenizer.eos_token_id)
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_seq_length:
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def main() -> None:
    args = parse_args()
    paths = cfg.setup_notebook_paths(blog_dir_name="1-distilling-a-0-8b-tool-calling-agent")
    user_model = cfg.required_env("TAU_BENCH_USER_SIMULATOR_LLM")
    user_slug = cfg.filename_slug(user_model)
    teacher_slug = cfg.filename_slug(args.teacher_model)
    student_slug = cfg.filename_slug(args.student_model)
    sft_path = Path(args.sft_path) if args.sft_path else (
        paths.output_dir / f"{teacher_slug}_{args.teacher_provider}_tau3_bench_retail_train_successful_sft_chat_rows_{user_slug}.jsonl"
    )
    work_dir = Path(args.output_dir) if args.output_dir else (
        paths.output_dir / f"{student_slug}_tau3_retail_sft_trl_peft"
    )
    data_dir = work_dir / "hf_tokenized_data"
    adapter_dir = work_dir / f"{student_slug}_tau3_retail_trl_lora_adapter"

    if not sft_path.exists():
        raise FileNotFoundError(f"SFT rows file not found: {sft_path}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = cfg.load_jsonl(sft_path)
    lengths = [sft_rows.mlx_chat_row_token_length(row, tokenizer) for row in rows]
    rows_that_fit = [row for row, length in zip(rows, lengths) if length <= args.max_seq_length]
    rows_too_long = len(rows) - len(rows_that_fit)
    if not rows_that_fit:
        raise RuntimeError("No SFT rows fit max sequence length.")

    train_rows, validation_rows, validation_task_ids = sft_rows.split_sft_rows_by_task_id(
        rows_that_fit,
        validation_task_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    if not validation_rows:
        validation_rows = train_rows[:1]
    train_examples = [
        example
        for row in train_rows
        if (example := tokenize_row(row, tokenizer, args.max_seq_length)) is not None
    ]
    validation_examples = [
        example
        for row in validation_rows
        if (example := tokenize_row(row, tokenizer, args.max_seq_length)) is not None
    ]
    if not train_examples:
        raise RuntimeError("No tokenized train examples fit max sequence length.")
    if not validation_examples:
        validation_examples = train_examples[:1]

    train_dataset = Dataset.from_list(train_examples)
    validation_dataset = Dataset.from_list(validation_examples)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.save_to_disk(data_dir / "train")
    validation_dataset.save_to_disk(data_dir / "valid")

    max_steps = args.max_steps
    if max_steps <= 0:
        max_steps = max(1, math.ceil(len(train_dataset) / max(1, args.batch_size * args.grad_accum)))
    save_steps = max(1, min(25, max_steps))
    eval_steps = max(1, min(25, max_steps))

    model_config = AutoConfig.from_pretrained(args.student_model, trust_remote_code=True)
    layer_count = int(model_config.num_hidden_layers)
    lora_layers = list(range(layer_count - args.lora_layers, layer_count))
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.student_model, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        layers_to_transform=lora_layers,
        layers_pattern="layers",
        bias="none",
    )
    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        max_length=args.max_seq_length,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=args.training_seed,
        data_seed=args.training_seed,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=data_collator,
    )
    resume_checkpoint = None
    if args.resume == "latest" and adapter_dir.exists():
        resume_checkpoint = get_last_checkpoint(str(adapter_dir))
    elif args.resume:
        resume_checkpoint = args.resume

    print("Student model:", args.student_model)
    print("SFT rows:", sft_path)
    print("Rows kept:", len(rows_that_fit), "dropped:", rows_too_long)
    print("Train examples:", len(train_dataset), "validation examples:", len(validation_dataset))
    print("Validation task ids:", sorted(validation_task_ids))
    print("Max steps:", max_steps)
    print("Batch size:", args.batch_size)
    print("LoRA layers:", lora_layers)
    print("LoRA target modules:", target_modules)
    print("Adapter dir:", adapter_dir)
    print("Resume checkpoint:", resume_checkpoint)

    train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metadata = {
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "teacher_provider": args.teacher_provider,
        "sft_path": str(sft_path),
        "adapter_dir": str(adapter_dir),
        "raw_rows": len(rows),
        "rows_kept": len(rows_that_fit),
        "rows_dropped_over_length": rows_too_long,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "max_steps": max_steps,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_layers": lora_layers,
        "lora_target_modules": target_modules,
        "metrics": train_output.metrics,
        "log_history": trainer.state.log_history,
    }
    metadata_path = work_dir / "training_metadata_trl_peft.json"
    metadata_path.write_text(json.dumps(cfg.make_json_safe(metadata), indent=2), encoding="utf-8")
    print("Saved adapter:", adapter_dir)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
