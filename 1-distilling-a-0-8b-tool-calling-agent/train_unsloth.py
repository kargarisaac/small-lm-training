from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import nestful


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NESTFUL with Unsloth + TRL.")
    parser.add_argument("--model", default=cfg.UNSLOTH_STUDENT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR / "nestful")
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from datasets import Dataset
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq

    rows = cfg.read_jsonl(args.train_path, args.limit) if args.train_path else nestful.load_prepared_rows(args.data_dir, "train", args.limit)
    train_rows, valid_rows = nestful.split_train_validation(rows, args.validation_fraction)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_examples = [example for row in train_rows if (example := nestful.tokenize_sft_row(tokenizer, row, args.max_seq_length))]
    valid_examples = [example for row in valid_rows if (example := nestful.tokenize_sft_row(tokenizer, row, args.max_seq_length))]
    if not train_examples:
        raise RuntimeError("No train rows fit max sequence length.")
    if not valid_examples:
        valid_examples = train_examples[:1]
    train_dataset = Dataset.from_list(train_examples)
    valid_dataset = Dataset.from_list(valid_examples)
    max_steps = args.max_steps if args.max_steps > 0 else math.ceil(len(train_dataset) / (args.batch_size * args.grad_accum))
    work_dir = args.output_dir or cfg.OUTPUT_DIR / f"{cfg.filename_slug(args.model)}_nestful_unsloth"
    adapter_dir = work_dir / "adapter"

    print("Model:", args.model)
    print("Raw rows:", len(rows))
    print("Train examples:", len(train_dataset))
    print("Validation examples:", len(valid_dataset))
    print("Max steps:", max_steps)
    print("Adapter dir:", adapter_dir)
    if args.dry_run:
        return

    try:
        from unsloth import FastLanguageModel
    except ImportError as error:
        raise RuntimeError("Install Unsloth on a CUDA/Linux machine before running this script: uv pip install unsloth") from error
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )
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
            eval_strategy="steps",
            eval_steps=max(1, min(100, max_steps)),
            save_strategy="steps",
            save_steps=max(1, min(100, max_steps)),
            save_total_limit=2,
            report_to=[],
            remove_unused_columns=False,
            packing=False,
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt"),
    )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    cfg.write_json(work_dir / "training_config.json", vars(args) | {"max_steps": max_steps, "adapter_dir": adapter_dir})
    print("Saved adapter:", adapter_dir)


if __name__ == "__main__":
    main()
