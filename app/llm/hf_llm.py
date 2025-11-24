"""
HF + LoRA バックエンド用の簡易LLMラッパ。
RAGPipeline から `.invoke(prompt)` で呼べるインターフェースを提供する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


@dataclass
class HFLLMConfig:
    """HFローカルLLMの設定"""
    model_name: str
    lora_path: Optional[str] = None
    device: str = "auto"  # "cuda" / "cpu" / "auto"
    max_new_tokens: int = 256
    temperature: float = 0.0
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"


class HFLoRALLM:
    """
    HFベースモデル＋（必要に応じて）LoRA重みを読み込み、invokeで生成するラッパ。
    """

    def __init__(self, config: HFLLMConfig) -> None:
        self.config = config

        # tokenizerロード
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            use_fast=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # モデルロード
        device_map = "auto" if config.device == "auto" else None
        if config.load_in_4bit:
            compute_dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype_map[config.bnb_4bit_compute_dtype],
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )

        # LoRA適用
        if config.lora_path:
            self.model = PeftModel.from_pretrained(self.model, config.lora_path)

        # 明示的にデバイスへ移動（auto以外指定時）
        if config.device != "auto":
            self.model = self.model.to(config.device)

        self.model.eval()

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """
        プロンプトを入力として生成を行い、生成部分のみ返す。
        kwargsで max_new_tokens / temperature を上書き可能。
        """
        max_new_tokens = kwargs.get("max_new_tokens", self.config.max_new_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        # モデルの実デバイスに合わせて入力を移動（device_map=auto時も含む）
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 生成テキスト全体から入力部を除去
        generated = outputs[0]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        # prompt長分を切り落とし（単純スライス）
        if decoded.startswith(prompt):
            return decoded[len(prompt):].strip()
        # fallback: そのまま返す
        return decoded.strip()
