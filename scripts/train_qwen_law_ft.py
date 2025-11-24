#!/usr/bin/env python3
"""
Qwen系モデルのLoRA/QLoRAファインチューニング用スクリプト。
Heavyな学習実行はここでは行わず、コード実装のみ。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


ASSISTANT_PREFIX = "<ASSISTANT>\n"
USER_PREFIX = "<USER>\n"


def build_sample_text(example: Dict[str, str]) -> str:
    """
    input/outputをシンプルに連結して学習テキストを作る。
    """
    return f"{USER_PREFIX}{example['input'].strip()}\n{ASSISTANT_PREFIX}{example['output'].strip()}"


def mask_input_tokens(labels, offsets, assistant_start: int):
    """
    offset_mappingを用いて、<ASSISTANT>より前のトークンを -100 でマスク。
    """
    if offsets is None or assistant_start < 0:
        return labels
    masked = labels.clone()
    for i, (_s, e) in enumerate(offsets):
        if e <= assistant_start:
            masked[i] = -100
    return masked


@dataclass
class TokenizeConfig:
    tokenizer: AutoTokenizer
    max_seq_length: int
    loss_on_output_only: bool


def tokenize_function(examples, cfg: TokenizeConfig):
    # batched入力（dict of list）を想定
    inputs = examples["input"]
    outputs = examples["output"]
    texts = [f"{USER_PREFIX}{inp.strip()}\n{ASSISTANT_PREFIX}{out.strip()}" for inp, out in zip(inputs, outputs)]
    enc = cfg.tokenizer(
        texts,
        max_length=cfg.max_seq_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=cfg.loss_on_output_only,
    )

    labels = enc["input_ids"]
    if cfg.loss_on_output_only:
        assistant_pos = [t.find(ASSISTANT_PREFIX) for t in texts]
        new_labels = []
        for i in range(len(labels)):
            lab_tensor = torch.tensor(labels[i])
            offs = enc["offset_mapping"][i]
            masked = mask_input_tokens(lab_tensor, offs, assistant_pos[i])
            new_labels.append(masked.tolist())
        enc["labels"] = new_labels
        enc.pop("offset_mapping", None)
    else:
        enc["labels"] = [l[:] for l in labels]
    return enc


def parse_args():
    p = argparse.ArgumentParser(description="Train Qwen LoRA on law QA JSONL")
    p.add_argument("--model-name", required=True, type=str, help="HFモデルID（例: Qwen/Qwen1.5-7B-Chat）")
    p.add_argument("--train-file", required=True, type=str, help="学習用JSONL")
    p.add_argument("--output-dir", required=True, type=str, help="出力ディレクトリ")

    # 学習設定
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    # 4bit設定
    p.add_argument("--use-4bit", action="store_true", help="bitsandbytes による4bit QLoRAを有効化する")
    p.add_argument("--bnb-4bit-compute-dtype", type=str, default="bfloat16",
                   choices=["float16", "bfloat16", "float32"],
                   help="4bit計算時のdtype")
    p.add_argument("--bnb-4bit-quant-type", type=str, default="nf4",
                   choices=["nf4", "fp4"],
                   help="4bit量子化の種類")

    # LoRA設定
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", type=str, default=None, help="カンマ区切りで指定。未指定なら自動推測に任せる。")
    p.add_argument("--lora-bias", type=str, default="none", choices=["none", "lora", "all"])

    # モード
    p.add_argument("--train-mode", type=str, default="auto", choices=["direct", "cot", "auto"], help="auto時はmeta.modeを利用")
    p.add_argument("--loss-on-output-only", action="store_true", help="可能であれば<ASSISTANT>以降のみ損失を計算")

    # 連絡・ログ
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=None)

    # 安全実行用フラグ/ステップ上書き
    p.add_argument("--do-train", action="store_true", help="指定時のみ trainer.train() を実行して保存する")
    p.add_argument("--max-steps-override", type=int, default=-1, help="TrainingArguments.max_steps を上書き（デフォルトは-1でエポック指定）")

    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_ds = load_dataset("json", data_files=args.train_file, split="train")

    # train-modeフィルタ（autoならそのまま、direct/cotならmeta.modeでフィルタ）
    if args.train_mode in ("direct", "cot"):
        def _flt(example):
            mode = example.get("meta", {}).get("mode")
            return mode == args.train_mode
        raw_ds = raw_ds.filter(_flt)

    tok_cfg = TokenizeConfig(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        loss_on_output_only=args.loss_on_output_only,
    )

    tokenized_ds = raw_ds.map(
        lambda ex: tokenize_function(ex, tok_cfg),
        batched=True,
        remove_columns=raw_ds.column_names,
    )

    # 4bit設定
    compute_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype_map[args.bnb_4bit_compute_dtype],
        )

    if args.use_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    target_modules: Optional[List[str]] = None
    if args.target_modules:
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        max_steps=args.max_steps_override,
        dataloader_num_workers=2,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 実行モード: --do-train 指定時のみ学習・保存を実行
    if args.do_train:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
    else:
        print("do-trainが指定されていないため、セットアップのみで終了します。")


if __name__ == "__main__":
    main()
