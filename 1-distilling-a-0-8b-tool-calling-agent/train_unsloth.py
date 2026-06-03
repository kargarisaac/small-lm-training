from __future__ import annotations

import argparse
import inspect
import math
import platform
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import sql_agent


def mlx_tune_forwards_grad_accumulation() -> bool:
    import mlx_tune.sft_trainer as sft_trainer

    source = inspect.getsource(sft_trainer.SFTTrainer)
    return "grad_accumulation_steps=self.gradient_accumulation_steps" in source


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SQL-agent student with Unsloth-style LoRA. Uses MLX-Tune on Apple Silicon and Unsloth on CUDA.")
    parser.add_argument("--backend", choices=["auto", "mlx", "cuda"], default="auto")
    parser.add_argument("--model", default=None)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=cfg.SFT_MAX_SEQ_LENGTH)
    parser.add_argument("--batch-size", type=int, default=cfg.SFT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=cfg.SFT_GRAD_ACCUM)
    parser.add_argument("--learning-rate", type=float, default=cfg.SFT_LEARNING_RATE)
    parser.add_argument("--lora-rank", type=int, default=cfg.SFT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=cfg.SFT_LORA_ALPHA)
    parser.add_argument("--target-modules", nargs="+", default=None)
    parser.add_argument("--experts-implementation", choices=["eager", "batched_mm", "grouped_mm"], default=None)
    parser.add_argument("--mlx-num-layers", type=int, default=cfg.SFT_MLX_NUM_LAYERS)
    parser.add_argument("--validation-fraction", type=float, default=cfg.SFT_VALIDATION_FRACTION)
    parser.add_argument("--seed", type=int, default=cfg.SFT_SEED)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-total-limit", type=int, default=6)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--mlx-subprocess", action="store_true", help="On Apple Silicon, train through MLX-Tune's mlx_lm.lora subprocess path.")
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.backend == "auto":
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            args.backend = "mlx"
        else:
            try:
                import torch

                args.backend = "cuda" if torch.cuda.is_available() else "mlx"
            except Exception:
                args.backend = "mlx"

    if args.mlx_subprocess and args.backend != "mlx":
        raise ValueError("--mlx-subprocess is only valid with --backend mlx")

    if args.model is None:
        args.model = cfg.MLX_STUDENT_MODEL if args.backend == "mlx" else cfg.UNSLOTH_STUDENT_MODEL

    mlx_grad_accumulation_is_forwarded = False
    if args.backend == "mlx":
        try:
            mlx_grad_accumulation_is_forwarded = mlx_tune_forwards_grad_accumulation()
        except Exception:
            mlx_grad_accumulation_is_forwarded = False

    if args.backend == "cuda" and not args.dry_run:
        try:
            from unsloth import FastLanguageModel, is_bfloat16_supported
        except ImportError as error:
            raise RuntimeError("Install Unsloth on the CUDA/Linux server first. Example: uv pip install unsloth") from error
    elif not args.dry_run:
        if not mlx_grad_accumulation_is_forwarded:
            raise RuntimeError(
                "Installed MLX-Tune does not forward gradient_accumulation_steps into MLX-LM. "
                "Patch mlx_tune/sft_trainer.py so TrainingArgs receives "
                "grad_accumulation_steps=self.gradient_accumulation_steps before running Mac training."
            )
        try:
            from mlx_tune import FastLanguageModel, SFTConfig, SFTTrainer, train_on_responses_only
        except ImportError as error:
            raise RuntimeError("Install MLX-Tune on this Mac first. Example: uv pip install mlx-tune") from error

    from datasets import Dataset
    from transformers import AutoConfig, AutoTokenizer, DataCollatorForSeq2Seq

    if args.train_path is None:
        raise ValueError("--train-path is required. Run notebook 02 to write the final filtered SFT file first.")
    rows = cfg.read_jsonl(args.train_path, args.limit)
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    target_modules = args.target_modules
    if target_modules is None:
        target_modules = cfg.SFT_LFM2_MOE_TARGET_MODULES if model_config.model_type == "lfm2_moe" else cfg.SFT_TARGET_MODULES
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepared = sql_agent.prepare_sft_rows(rows, tokenizer, args.max_seq_length, args.validation_fraction)
    train_examples = [example for row in prepared["train_rows"] if (example := sql_agent.tokenize_sft_row(tokenizer, row, args.max_seq_length))]
    valid_examples = [example for row in prepared["valid_rows"] if (example := sql_agent.tokenize_sft_row(tokenizer, row, args.max_seq_length))]
    if not train_examples:
        raise RuntimeError("No train rows fit max sequence length.")
    if not valid_examples:
        valid_examples = train_examples[:1]

    actual_grad_accum = args.grad_accum if args.backend != "mlx" or mlx_grad_accumulation_is_forwarded else 1
    if args.backend == "mlx":
        steps_per_epoch = math.ceil(len(train_examples) / args.batch_size)
        optimizer_steps_per_epoch = math.ceil(steps_per_epoch / actual_grad_accum)
    else:
        steps_per_epoch = math.ceil(len(train_examples) / (args.batch_size * actual_grad_accum))
        optimizer_steps_per_epoch = steps_per_epoch
    max_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch
    work_dir = (args.output_dir or cfg.OUTPUT_DIR / f"{cfg.filename_slug(args.model)}_sql_agent_unsloth").resolve()
    adapter_dir = work_dir / "adapter"

    print("Backend:", args.backend)
    print("Model:", args.model)
    for key, value in prepared["stats"].items():
        print(f"{key.replace('_', ' ').title()}:", value)
    print("Train examples:", len(train_examples))
    print("Validation examples:", len(valid_examples))
    print("Max training iterations:", max_steps)
    print("Training iterations per epoch:", steps_per_epoch)
    print("Estimated optimizer updates per epoch:", optimizer_steps_per_epoch)
    print("Gradient accumulation used:", actual_grad_accum)
    print("LoRA target modules:", target_modules)
    print("Experts implementation:", args.experts_implementation)
    if args.backend == "mlx":
        print("MLX-Tune forwards gradient accumulation:", mlx_grad_accumulation_is_forwarded)
        print("MLX subprocess training:", args.mlx_subprocess)
        print("MLX LoRA layers:", args.mlx_num_layers)
    print("Adapter dir:", adapter_dir)
    print("Load in 4bit:", args.load_in_4bit)
    print("Gradient checkpointing:", not args.no_grad_checkpoint)
    print("Validation during training:", not args.no_validation)
    if args.resume_from_checkpoint is not None:
        if not args.resume_from_checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {args.resume_from_checkpoint}")
        print("Resume checkpoint:", args.resume_from_checkpoint)
    if args.dry_run:
        return

    if args.backend == "mlx" and args.mlx_subprocess:
        model = SimpleNamespace(
            model_name=args.model,
            lora_config={"r": args.lora_rank, "lora_alpha": args.lora_alpha, "lora_dropout": 0.0},
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=args.load_in_4bit,
        )
        if args.experts_implementation is not None:
            expert_configs = {}
            for module in model.modules():
                config = getattr(module, "config", None)
                if config is not None and hasattr(config, "_experts_implementation"):
                    expert_configs[id(config)] = config
            for config in expert_configs.values():
                config._experts_implementation = args.experts_implementation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            target_modules=target_modules,
            bias="none",
            use_gradient_checkpointing=False if args.no_grad_checkpoint else "unsloth",
            random_state=args.seed,
        )

    if args.backend == "mlx":
        train_dataset = Dataset.from_list([{"messages": row["messages"]} for row in prepared["train_rows"]])
        valid_dataset = Dataset.from_list([{"messages": row["messages"]} for row in prepared["valid_rows"] or prepared["train_rows"][:1]])
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=None if args.no_validation else valid_dataset,
            args=SFTConfig(
                output_dir=str(work_dir / "mlx_trainer_data"),
                max_steps=max_steps,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=args.learning_rate,
                logging_steps=1,
                save_steps=max(1, min(100, max_steps)),
                max_seq_length=args.max_seq_length,
                grad_checkpoint=not args.no_grad_checkpoint,
                num_layers=args.mlx_num_layers,
                val_batches=0 if args.no_validation else 5,
                steps_per_eval=max(1, min(200, max_steps)),
                use_native_training=not args.mlx_subprocess,
            ),
            adapter_path=str(adapter_dir.resolve()),
            lora_r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
        )
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
    else:
        train_dataset = Dataset.from_list(train_examples)
        valid_dataset = Dataset.from_list(valid_examples)
        from trl import SFTConfig, SFTTrainer

        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(
                output_dir=str(adapter_dir),
                max_length=args.max_seq_length,
                max_steps=max_steps,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=args.learning_rate,
                logging_steps=1,
                eval_strategy="no" if args.no_validation else "steps",
                eval_steps=max(1, min(100, max_steps)),
                save_strategy="steps",
                save_steps=max(1, min(100, max_steps)),
                save_total_limit=args.save_total_limit,
                bf16=is_bfloat16_supported(),
                fp16=not is_bfloat16_supported(),
                gradient_checkpointing=not args.no_grad_checkpoint,
                report_to=[],
                remove_unused_columns=False,
                packing=False,
                dataset_kwargs={"skip_prepare_dataset": True},
                seed=args.seed,
            ),
            train_dataset=train_dataset,
            eval_dataset=None if args.no_validation else valid_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt"),
        )
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    if args.backend == "cuda":
        trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    cfg.write_json(
        work_dir / "training_config.json",
        vars(args)
        | {
            "max_steps": max_steps,
            "training_iterations_per_epoch": steps_per_epoch,
            "estimated_optimizer_updates_per_epoch": optimizer_steps_per_epoch,
            "actual_grad_accum": actual_grad_accum,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "experts_implementation": args.experts_implementation,
            "target_modules": target_modules,
            "adapter_dir": adapter_dir,
            "stats": prepared["stats"],
        },
    )
    print("Saved adapter:", adapter_dir)


if __name__ == "__main__":
    main()
