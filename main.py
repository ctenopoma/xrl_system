"""XRL System - CLI エントリーポイント。

JSBSim の軌跡ログ CSV から LLM ベースの XRL 分析を実行する。

使い方:
    python main.py --method sysllm --evaluate
    python main.py --method cot --step 150 --evaluate
    python main.py --method mcts --step 150 --evaluate
    python main.py --method talktoagent --query "Step 150でなぜスロットルを下げたのか"
"""

from __future__ import annotations

import argparse
import sys

from modules.data_loader import DataLoader
from modules.evaluator import EpisodeEvaluator
from modules.llm_client import LLMClient
from modules.mcts_xrl import MCTSXRL
from modules.sysllm import SySLLM
from modules.talktoagent import TalkToAgent


# ------------------------------------------------------------------
# CLI パーサー定義
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="XRL System - LLM-based Explainable RL Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
実行例:
  python main.py --method sysllm --evaluate
  python main.py --method cot --step 150 --evaluate
  python main.py --method mcts --step 150 --iterations 6 --evaluate
  python main.py --method talktoagent --query "Step 150でなぜスロットルを下げたのか"

環境変数:
  XRL_MODEL_NAME  使用するLLMモデル名 (デフォルト: gpt-4o)
  XRL_API_KEY     APIキー
""",
    )
    p.add_argument(
        "--method",
        choices=["cot", "sysllm", "talktoagent", "mcts"],
        required=True,
        help="実行する分析手法",
    )
    p.add_argument(
        "--csv",
        default="data/trajectory_log.csv",
        help="軌跡ログCSVのパス (デフォルト: data/trajectory_log.csv)",
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        help="cot/mcts で分析するステップ番号 (必須)",
    )
    p.add_argument(
        "--query",
        type=str,
        default=None,
        help="talktoagent へのユーザー質問 (必須)",
    )
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="生成説明を LLM-as-a-Judge で自動評価する",
    )
    p.add_argument(
        "--model",
        default=None,
        help="使用する LLM モデル名 (省略時は環境変数 XRL_MODEL_NAME または gpt-4o)",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=4,
        help="MCTS の探索回数 (デフォルト: 4)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=50,
        dest="max_rows",
        help="sysllm で LLM に渡す最大行数 (デフォルト: 50)",
    )
    return p


# ------------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """引数に応じて対応モジュールを実行しコンソール出力する。"""
    # 共通コンポーネントを初期化
    llm = LLMClient(model=args.model)
    loader = DataLoader(csv_path=args.csv)
    loader.load()

    result: dict = {}

    # --- 各モード ---
    if args.method == "sysllm":
        print("=" * 60)
        print("[SySLLM] エピソード全体の戦術を分析中...")
        print("=" * 60)
        result = SySLLM(llm, loader).analyze(max_rows=args.max_rows)
        _print_sysllm_result(result)

    elif args.method == "cot":
        _require_step(args)
        print("=" * 60)
        print(f"[CoT] Step {args.step} の行動理由を分析中...")
        print("=" * 60)
        result = MCTSXRL(llm, loader).explain_cot(args.step)
        _print_local_result("CoT", result)

    elif args.method == "mcts":
        _require_step(args)
        print("=" * 60)
        print(f"[MCTS-XRL] Step {args.step} の行動理由を反復改善中 (iterations={args.iterations})...")
        print("=" * 60)
        result = MCTSXRL(llm, loader, iterations=args.iterations).explain_mcts(args.step)
        _print_mcts_result(result)

    elif args.method == "talktoagent":
        if not args.query:
            _exit_error("--query が必要です (--method talktoagent)")
        print("=" * 60)
        print("[TalkToAgent] データドリブン分析中...")
        print("=" * 60)
        result = TalkToAgent(llm, loader).answer(args.query)
        _print_talktoagent_result(result)

    # --- 自動評価 ---
    if args.evaluate:
        print("\n" + "=" * 60)
        print("[Evaluator] LLM-as-a-Judge による自動評価中...")
        print("=" * 60)
        judge = EpisodeEvaluator(llm)
        ctx = loader.get_step_context(args.step) if args.step else None
        traj = loader.to_trajectory_text() if args.method == "sysllm" else ""
        eval_result = judge.evaluate(result["explanation"], ctx, traj)
        _print_eval(eval_result)


# ------------------------------------------------------------------
# 出力ヘルパー
# ------------------------------------------------------------------

def _print_sysllm_result(result: dict) -> None:
    print(f"\n[使用キーフレーム数: {result['n_keyframes']}]\n")
    labels = {
        "tactical_approach":      "1. 戦術的アプローチ",
        "situational_adaptation": "2. 状況への適応",
        "inefficiencies":         "3. 非効率性と弱点",
        "overall_summary":        "4. 総合要約",
    }
    for key, label in labels.items():
        print(f"\n## {label}")
        print(result.get(key, "(取得失敗)"))


def _print_local_result(method: str, result: dict) -> None:
    ctx = result["context"]
    print(f"\n[Step {result['step']}]")
    print(f"状態: {ctx['state']}")
    print(f"操舵: {ctx['action']}")
    print(f"\n【{method} 説明】")
    print(result["explanation"])


def _print_mcts_result(result: dict) -> None:
    _print_local_result("MCTS-XRL", result)
    print(f"\n[最高 Q 値: {result['best_q']:.3f}]")
    print("\n[ツリーサマリー]")
    for i, node_info in enumerate(result["tree_summary"]):
        print(
            f"  ノード{i}: Q={node_info['q_value']:.3f}, "
            f"訪問={node_info['visits']}, "
            f"説明={node_info['explanation_snippet']}"
        )


def _print_talktoagent_result(result: dict) -> None:
    print(f"\n【質問】 {result['query']}")
    print(f"\n【Coordinator 計画】\n{result['plan']}")
    print(f"\n【実行コード】\n{result['code']}")
    print(f"\n【実行結果】\n{result['exec_output']}")
    print(f"\n【Explainer 回答】\n{result['explanation']}")
    if result["retries"] > 0:
        print(f"\n[デバッグ再試行: {result['retries']} 回]")


def _print_eval(eval_result: dict) -> None:
    soundness = eval_result.get("soundness", "?")
    fidelity = eval_result.get("fidelity", "?")
    reason = eval_result.get("reason", "")
    total = (soundness if isinstance(soundness, int) else 0) + \
            (fidelity if isinstance(fidelity, int) else 0)
    print(f"\n【評価結果】")
    print(f"  Soundness (論理的妥当性): {soundness}/2")
    print(f"  Fidelity  (忠実性):       {fidelity}/2")
    print(f"  合計スコア:               {total}/4")
    print(f"  理由: {reason}")


# ------------------------------------------------------------------
# バリデーションヘルパー
# ------------------------------------------------------------------

def _require_step(args: argparse.Namespace) -> None:
    if args.step is None:
        _exit_error(f"--step が必要です (--method {args.method})")


def _exit_error(message: str) -> None:
    print(f"エラー: {message}", file=sys.stderr)
    sys.exit(1)


# ------------------------------------------------------------------
# エントリーポイント
# ------------------------------------------------------------------

if __name__ == "__main__":
    run(build_parser().parse_args())
