"""LoRA 学習スクリプト。

DatasetBuilder が生成したデータセットを使って、ベースモデルを LoRA でファインチューニングする。
学習済みアダプタは models/<run_id>/adapter/ に保存される。

使い方:
    # 基本実行
    python train_lora.py --dataset datasets/mcts_v1basic_score30_20260324_123456

    # rank と epoch を指定
    python train_lora.py --dataset datasets/... --rank 8 --epochs 3

    # 4-bit 量子化でメモリを節約 (要 bitsandbytes)
    python train_lora.py --dataset datasets/... --load-in-4bit

    # 学習後に評価を実行
    python train_lora.py --dataset datasets/... --eval-after

    # W&B ロギングを有効化
    python train_lora.py --dataset datasets/... --wandb

必要なパッケージ:
    pip install transformers peft trl accelerate datasets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from modules.lora_trainer import LoRAConfig, LoRATrainer, MODELS_DIR


# ------------------------------------------------------------------
# CLI パーサー
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LoRA ファインチューニング",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="DatasetBuilder が生成したデータセットディレクトリのパス",
    )
    p.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        dest="base_model",
        help="ベースモデルの HuggingFace ID またはローカルパス (デフォルト: Qwen/Qwen2.5-7B-Instruct)",
    )
    p.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA の rank (r) (デフォルト: 16)",
    )
    p.add_argument(
        "--alpha",
        type=int,
        default=None,
        help="LoRA の alpha (デフォルト: rank × 2)",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="LoRA ドロップアウト率 (デフォルト: 0.05)",
    )
    p.add_argument(
        "--target-modules",
        default="all-linear",
        dest="target_modules",
        help='LoRA を適用するモジュール名 (デフォルト: "all-linear")',
    )
    p.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="学習率 (デフォルト: 2e-4)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="エポック数 (デフォルト: 3)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        dest="batch_size",
        help="1GPU あたりのバッチサイズ (デフォルト: 4)",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        dest="grad_accum",
        help="勾配蓄積ステップ数 (デフォルト: 4)",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=1024,
        dest="max_length",
        help="最大シーケンス長 (トークン数, デフォルト: 1024)",
    )
    p.add_argument(
        "--load-in-4bit",
        action="store_true",
        dest="load_in_4bit",
        help="bitsandbytes 4-bit 量子化を使用 (GPU メモリ節約)",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="float16 で学習 (use_bf16 が False の場合に有効)",
    )
    p.add_argument(
        "--no-bf16",
        action="store_true",
        dest="no_bf16",
        help="bfloat16 を無効化 (CPU や旧 GPU 向け)",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="W&B ロギングを有効化",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (デフォルト: 42)",
    )
    p.add_argument(
        "--output-dir",
        default=str(MODELS_DIR),
        dest="output_dir",
        help=f"モデルの保存先ディレクトリ (デフォルト: {MODELS_DIR})",
    )
    p.add_argument(
        "--eval-after",
        action="store_true",
        dest="eval_after",
        help="学習後に evaluate_baseline.py で評価を実行する",
    )
    return p


# ------------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"[エラー] データセットディレクトリが見つかりません: {dataset_dir}")
        sys.exit(1)

    alpha = args.alpha if args.alpha is not None else args.rank * 2

    cfg = LoRAConfig(
        base_model=args.base_model,
        lora_rank=args.rank,
        lora_alpha=alpha,
        lora_dropout=args.dropout,
        target_modules=args.target_modules,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        load_in_4bit=args.load_in_4bit,
        use_bf16=not args.no_bf16,
        use_fp16=args.fp16,
        use_wandb=args.wandb,
        seed=args.seed,
    )

    # 学習設定をサマリー表示
    print("=" * 60)
    print("  LoRA 学習設定")
    print("=" * 60)
    print(f"  base_model  : {cfg.base_model}")
    print(f"  dataset     : {dataset_dir}")
    print(f"  rank/alpha  : {cfg.lora_rank}/{cfg.lora_alpha}")
    print(f"  lr          : {cfg.learning_rate}")
    print(f"  epochs      : {cfg.num_epochs}")
    print(f"  batch       : {cfg.per_device_train_batch_size} × {cfg.gradient_accumulation_steps} (accum)")
    print(f"  max_length  : {cfg.max_length}")
    print(f"  load_in_4bit: {cfg.load_in_4bit}")
    print(f"  bf16        : {cfg.use_bf16}")
    print("=" * 60)

    trainer = LoRATrainer(cfg)
    run_id = trainer.train(dataset_dir)

    output_dir = Path(args.output_dir) / run_id
    adapter_dir = output_dir / "adapter"

    print(f"\n{'=' * 60}")
    print(f"  run_id   : {run_id}")
    print(f"  アダプタ : {adapter_dir}")
    print(f"\n  次のステップ:")
    print(f"    python evaluate_baseline.py --sample 20 \\")
    print(f"        --backend local --adapter {adapter_dir}")
    print("=" * 60)

    # --eval-after フラグが付いている場合は evaluate_baseline.py を自動実行
    if args.eval_after:
        _run_eval_after(adapter_dir)


def _run_eval_after(adapter_dir: Path) -> None:
    """学習後に evaluate_baseline.py を実行する。"""
    import subprocess

    cmd = [
        sys.executable, "evaluate_baseline.py",
        "--sample", "10",
        "--backend", "local",
        "--adapter", str(adapter_dir),
    ]
    print(f"\n[eval-after] 実行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[警告] evaluate_baseline.py が終了コード {result.returncode} で終了しました")


# ------------------------------------------------------------------
# エントリーポイント
# ------------------------------------------------------------------

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run(build_parser().parse_args())
