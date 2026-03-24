"""学習データセット生成スクリプト。

XRL手法 (CoT / MCTS / SySLLM) をラベルソースとして使い、
LoRA 学習用の (instruction, input, output) JSONL データセットを生成する。

使い方:
    # CoT で全ステップ、フィルタなし
    python build_dataset.py --label-source cot

    # MCTS で全ステップ、スコア 3.0 以上を収集
    python build_dataset.py --label-source mcts --min-score 3.0

    # CoT で特定ステップのみ (動作確認用)
    python build_dataset.py --label-source cot --steps 50 100 150

    # SySLLM 要約を事前情報として注入 (v2_with_prior テンプレート推奨)
    python build_dataset.py --label-source sysllm --template v2_with_prior

    # テンプレートと MCTS イテレーション数を指定
    python build_dataset.py --label-source mcts --template v1_combat_only \\
        --mcts-iterations 6 --min-score 2.0

環境変数:
    XRL_MODEL_NAME  使用する LLM モデル名 (デフォルト: gpt-4o)
    XRL_API_KEY     API キー
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from modules.data_loader import DataLoader
from modules.dataset_builder import DatasetBuilder, LabelSourceConfig, DATASETS_DIR
from modules.evaluator import EpisodeEvaluator
from modules.llm_client import LLMClient
from modules.prompt_template import PromptTemplate, PRESETS


# ------------------------------------------------------------------
# CLI パーサー
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LoRA 学習用データセット生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--label-source",
        choices=["cot", "mcts", "sysllm"],
        default="cot",
        dest="label_source",
        help="ラベル生成手法 (デフォルト: cot)",
    )
    p.add_argument(
        "--template",
        choices=list(PRESETS.keys()),
        default="v1_basic",
        help="プロンプトテンプレート (デフォルト: v1_basic)",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        dest="min_score",
        help="品質フィルタ: Soundness+Fidelity の最低スコア 0〜4 (デフォルト: 0.0 = フィルタなし)",
    )
    p.add_argument(
        "--mcts-iterations",
        type=int,
        default=4,
        dest="mcts_iterations",
        help="MCTS のイテレーション数 (デフォルト: 4, label-source=mcts 時のみ有効)",
    )
    p.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="生成対象のステップ番号 (省略時は全ステップ)",
    )
    p.add_argument(
        "--csv",
        default="data/trajectory_log.csv",
        help="軌跡ログ CSV (デフォルト: data/trajectory_log.csv)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="LLM モデル名 (省略時は XRL_MODEL_NAME 環境変数または gpt-4o)",
    )
    p.add_argument(
        "--output-dir",
        default=str(DATASETS_DIR),
        dest="output_dir",
        help=f"データセットの保存先ディレクトリ (デフォルト: {DATASETS_DIR})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="train/val 分割の乱数シード (デフォルト: 42)",
    )
    return p


# ------------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # バリデーション
    if args.min_score < 0.0 or args.min_score > 4.0:
        print(f"[エラー] --min-score は 0.0〜4.0 の範囲で指定してください: {args.min_score}")
        sys.exit(1)

    if args.label_source == "sysllm" and args.template != "v2_with_prior":
        print(
            "[警告] --label-source sysllm は --template v2_with_prior と組み合わせることを推奨します。"
            f" (現在: {args.template})"
        )

    llm = LLMClient(model=args.model)
    loader = DataLoader(csv_path=args.csv)
    loader.load()  # 早期にデータをロード (ダミー生成含む)

    label_cfg = LabelSourceConfig(
        method=args.label_source,
        min_score=args.min_score,
        mcts_iterations=args.mcts_iterations,
    )
    template = PromptTemplate.from_preset(args.template)
    evaluator = EpisodeEvaluator(llm)

    builder = DatasetBuilder(
        llm=llm,
        loader=loader,
        template=template,
        label_config=label_cfg,
        evaluator=evaluator,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    dataset_id = builder.build(steps=args.steps)

    print(f"\n{'=' * 60}")
    print(f"  dataset_id : {dataset_id}")
    print(f"  保存先     : {Path(args.output_dir) / dataset_id}")
    print(f"  次のステップ: python train_lora.py --dataset {Path(args.output_dir) / dataset_id}")
    print("=" * 60)


# ------------------------------------------------------------------
# エントリーポイント
# ------------------------------------------------------------------

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run(build_parser().parse_args())
