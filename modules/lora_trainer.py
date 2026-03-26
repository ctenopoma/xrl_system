"""LoRA Trainer — HuggingFace PEFT + TRL SFTTrainer によるファインチューニング。

DatasetBuilder が生成した JSONL データセットを読み込み、
LoRA でベースモデルをファインチューニングする。
学習済みアダプタは models/<run_id>/adapter/ に保存される。

依存パッケージ:
    pip install transformers peft trl accelerate datasets

使い方 (コードから):
    from modules.lora_trainer import LoRAConfig, LoRATrainer

    cfg = LoRAConfig(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_rank=16,
        num_epochs=3,
    )
    trainer = LoRATrainer(cfg)
    run_id = trainer.train("datasets/mcts_v1basic_score30_20260324_123456")
    # → models/<run_id>/adapter/ に保存
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

MODELS_DIR = Path("models")


# ------------------------------------------------------------------
# 設定データクラス
# ------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """LoRA 学習設定。

    Attributes:
        base_model:                  HuggingFace モデル ID またはローカルパス
        lora_rank:                   LoRA の rank (r)。大きいほど表現力↑・メモリ↑
        lora_alpha:                  LoRA の alpha (通常 rank の 2 倍が目安)
        lora_dropout:                LoRA ドロップアウト率
        target_modules:              LoRA を適用するモジュール。
                                     "all-linear" で全 Linear 層に適用
        learning_rate:               学習率
        num_epochs:                  エポック数
        per_device_train_batch_size: 1GPU あたりのバッチサイズ
        gradient_accumulation_steps: 勾配蓄積ステップ数
                                     (実効バッチ = per_device × grad_accum × GPU数)
        max_length:                  最大シーケンス長 (トークン数)
        warmup_ratio:                ウォームアップ比率 (全ステップ中)
        lr_scheduler_type:           学習率スケジューラ ("cosine", "linear" など)
        load_in_4bit:                bitsandbytes 4bit 量子化を使うか
                                     (GPU メモリ節約。bitsandbytes が必要)
        use_bf16:                    bfloat16 を使うか (Ampere 以降の GPU 推奨)
        use_fp16:                    float16 を使うか (use_bf16 が False の場合に有効)
        use_wandb:                   W&B ロギングを有効にするか
        seed:                        乱数シード
    """

    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    load_in_4bit: bool = False
    use_bf16: bool = True
    use_fp16: bool = False
    use_wandb: bool = False
    seed: int = 42


# ------------------------------------------------------------------
# LoRATrainer 本体
# ------------------------------------------------------------------

class LoRATrainer:
    """DatasetBuilder 出力の JSONL を使って LoRA ファインチューニングを行う。

    使い方::

        cfg = LoRAConfig(base_model="Qwen/Qwen2.5-7B-Instruct", lora_rank=16)
        trainer = LoRATrainer(cfg)
        run_id = trainer.train("datasets/mcts_v1basic_score30_20260324_123456")
    """

    def __init__(self, config: Optional[LoRAConfig] = None) -> None:
        self.config = config or LoRAConfig()

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def train(self, dataset_dir: str | Path) -> str:
        """学習を実行してアダプタを保存し、run_id を返す。

        Args:
            dataset_dir: DatasetBuilder が生成したデータセットディレクトリ

        Returns:
            run_id: models/<run_id>/ に保存された識別子

        Raises:
            FileNotFoundError: dataset_dir が存在しない場合
            ImportError:       必要なパッケージがインストールされていない場合
        """
        self._check_imports()

        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"データセットディレクトリが見つかりません: {dataset_dir}")

        run_id = self._make_run_id(dataset_dir)
        output_dir = MODELS_DIR / run_id
        adapter_dir = output_dir / "adapter"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[LoRATrainer] run_id={run_id}")
        print(f"[LoRATrainer] base_model={self.config.base_model}")
        print(f"[LoRATrainer] dataset={dataset_dir}")

        # W&B 設定
        if not self.config.use_wandb:
            os.environ.setdefault("WANDB_DISABLED", "true")

        # --- 1. データセット読み込み ---
        train_ds, val_ds = self._load_datasets(dataset_dir)
        print(f"[LoRATrainer] train={len(train_ds)} | val={len(val_ds)}")

        # --- 2. モデル & トークナイザー ---
        tokenizer, model = self._load_model()

        # --- 3. LoRA 適用 ---
        from peft import LoraConfig, get_peft_model, TaskType

        target = (
            self.config.target_modules.split(",")
            if "," in self.config.target_modules
            else self.config.target_modules  # "all-linear" or list
        )
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target,
            bias="none",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

        # --- 4. SFTTrainer ---
        from trl import SFTConfig, SFTTrainer

        sft_cfg = SFTConfig(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            max_length=self.config.max_length,
            bf16=self.config.use_bf16,
            fp16=self.config.use_fp16,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.config.seed,
            report_to="wandb" if self.config.use_wandb else "none",
            run_name=run_id if self.config.use_wandb else None,
        )

        # フォーマット関数: サンプル → 学習用テキスト
        def formatting_func(sample: dict) -> str:
            return _format_sample(sample, tokenizer)

        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            formatting_func=formatting_func,
        )

        print("[LoRATrainer] 学習開始...")
        train_result = trainer.train()

        # --- 5. アダプタ & トークナイザー保存 ---
        print(f"[LoRATrainer] アダプタを保存中: {adapter_dir}")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        # --- 6. run_config.json 保存 ---
        final_loss = (
            train_result.training_loss
            if hasattr(train_result, "training_loss")
            else None
        )
        self._save_run_config(run_id, dataset_dir, output_dir, final_loss)

        print(f"\n[LoRATrainer] 完了 run_id={run_id}")
        print(f"  アダプタ: {adapter_dir}")
        if final_loss is not None:
            print(f"  最終学習ロス: {final_loss:.4f}")

        return run_id

    # ------------------------------------------------------------------
    # 非公開ヘルパー
    # ------------------------------------------------------------------

    @staticmethod
    def _check_imports() -> None:
        """必要なパッケージの有無を確認する。"""
        missing = []
        for pkg in ("transformers", "peft", "trl", "datasets", "accelerate"):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            raise ImportError(
                f"以下のパッケージが必要です: {missing}\n"
                "pip install transformers peft trl accelerate datasets"
            )

    def _load_datasets(self, dataset_dir: Path):
        """train.jsonl / val.jsonl を HuggingFace Dataset として読み込む。"""
        from datasets import load_dataset

        train_ds = load_dataset(
            "json",
            data_files=str(dataset_dir / "train.jsonl"),
            split="train",
        )
        val_ds = load_dataset(
            "json",
            data_files=str(dataset_dir / "val.jsonl"),
            split="train",
        )
        return train_ds, val_ds

    def _load_model(self):
        """トークナイザーとモデルを読み込む。"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        cfg = self.config
        print(f"[LoRATrainer] トークナイザーを読み込み中: {cfg.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # 量子化設定
        bnb_config = None
        if cfg.load_in_4bit:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as e:
                print(f"[警告] 4bit 量子化の設定に失敗しました ({e})。通常精度で読み込みます。")

        # dtype 決定
        if bnb_config is not None:
            dtype = None  # BnB が管理
        elif cfg.use_bf16 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif cfg.use_fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32

        print(f"[LoRATrainer] モデルを読み込み中: {cfg.base_model} (dtype={dtype})")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.use_cache = False  # gradient checkpointing との互換性

        return tokenizer, model

    @staticmethod
    def _make_run_id(dataset_dir: Path) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{dataset_dir.name[:40]}_{ts}"

    def _save_run_config(
        self,
        run_id: str,
        dataset_dir: Path,
        output_dir: Path,
        final_loss: Optional[float],
    ) -> None:
        config_dict = asdict(self.config)
        config_dict.update({
            "run_id":      run_id,
            "dataset_dir": str(dataset_dir),
            "adapter_dir": str(output_dir / "adapter"),
            "trained_at":  datetime.now().isoformat(),
        })
        if final_loss is not None:
            config_dict["final_train_loss"] = round(final_loss, 6)

        path = output_dir / "run_config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        print(f"[LoRATrainer] 設定を保存: {path}")


# ------------------------------------------------------------------
# プロンプトフォーマット (モジュールレベル関数: テスト・再利用に便利)
# ------------------------------------------------------------------

def _format_sample(sample: dict, tokenizer) -> str:
    """1サンプルを学習用テキストにフォーマットする。

    優先順位:
    1. tokenizer.apply_chat_template() (チャットテンプレートがある場合)
    2. シンプルなフォールバック形式

    Args:
        sample:    {"instruction": str, "input": str, "output": str, ...}
        tokenizer: HuggingFace Tokenizer インスタンス

    Returns:
        学習用テキスト文字列
    """
    messages = [
        {"role": "system",    "content": sample["instruction"]},
        {"role": "user",      "content": sample["input"]},
        {"role": "assistant", "content": sample["output"]},
    ]
    try:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
    except Exception:
        pass

    # フォールバック: シンプルな区切り形式
    return (
        f"### System:\n{sample['instruction']}\n\n"
        f"### User:\n{sample['input']}\n\n"
        f"### Assistant:\n{sample['output']}"
    )
