"""XRL System - CLI エントリーポイント。

JSBSim の軌跡ログ CSV から LLM ベースの XRL 分析を実行する。

使い方:
    python main.py --method sysllm --evaluate
    python main.py --method cot --step 150 --evaluate
    python main.py --method mcts --step 150 --evaluate
    python main.py --method talktoagent --query "Step 150でなぜスロットルを下げたのか"
    python main.py --method compare --step 150
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# Windows の cp932 による UnicodeEncodeError を防ぐ
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
  python main.py --method compare --step 150
  python main.py --method compare --step 150 --output results/compare_step150.json

環境変数:
  XRL_MODEL_NAME  使用するLLMモデル名 (デフォルト: gpt-4o)
  XRL_API_KEY     APIキー
""",
    )
    p.add_argument(
        "--method",
        choices=["cot", "sysllm", "talktoagent", "mcts", "compare"],
        required=True,
        help="実行する分析手法 (compare: CoT/SySLLM/MCTSを一括実行して比較)",
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
        help="cot/mcts/compare で分析するステップ番号",
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
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="compare の結果を保存する JSON ファイルパス",
    )
    p.add_argument(
        "--query",
        type=str,
        default=None,
        help="talktoagent / compare でのユーザー質問 (省略時はデフォルト質問を使用)",
    )
    return p


# ------------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """引数に応じて対応モジュールを実行しコンソール出力する。"""
    llm    = LLMClient(model=args.model)
    loader = DataLoader(csv_path=args.csv)
    loader.load()

    if args.method == "compare":
        _run_compare(args, llm, loader)
        return

    result: dict = {}

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
        if not args.query and args.step is None:
            _exit_error("--query または --step が必要です (--method talktoagent)")
        if not args.query:
            args.query = _DEFAULT_QUERY_TEMPLATE.format(step=args.step)
        print("=" * 60)
        print("[TalkToAgent] データドリブン分析中...")
        print("=" * 60)
        result = TalkToAgent(llm, loader).answer(args.query)
        _print_talktoagent_result(result)

    if args.evaluate:
        print("\n" + "=" * 60)
        print("[Evaluator] LLM-as-a-Judge による自動評価中...")
        print("=" * 60)
        judge   = EpisodeEvaluator(llm)
        ctx     = loader.get_step_context(args.step) if args.step else None
        traj    = loader.to_trajectory_text() if args.method == "sysllm" else ""
        eval_r  = judge.evaluate(result["explanation"], ctx, traj)
        _print_eval(eval_r)


# ------------------------------------------------------------------
# compare モード
# ------------------------------------------------------------------

_DEFAULT_QUERY_TEMPLATE = "Step {step} でどのような戦術的判断をしたか、操舵入力の数値を根拠に説明して"


def _run_compare(args: argparse.Namespace, llm: LLMClient, loader: DataLoader) -> None:
    """CoT / SySLLM / MCTS / TalkToAgent を一括実行して評価スコアを比較する。"""
    if args.step is None:
        _exit_error("--step が必要です (--method compare)")

    query    = args.query or _DEFAULT_QUERY_TEMPLATE.format(step=args.step)
    judge    = EpisodeEvaluator(llm)
    ctx      = loader.get_step_context(args.step)
    traj     = loader.to_trajectory_text()
    results  = {}

    # --- 1. CoT ---
    print("=" * 60)
    print(f"[1/4] CoT: Step {args.step} を分析中...")
    print("=" * 60)
    cot_result = MCTSXRL(llm, loader).explain_cot(args.step)
    _print_local_result("CoT", cot_result)
    cot_eval = judge.evaluate(cot_result["explanation"], ctx)
    _print_eval(cot_eval)
    results["cot"] = {**cot_result, "eval": cot_eval}

    # --- 2. SySLLM ---
    print("\n" + "=" * 60)
    print("[2/4] SySLLM: エピソード全体を分析中...")
    print("=" * 60)
    sys_result = SySLLM(llm, loader).analyze(max_rows=args.max_rows)
    _print_sysllm_result(sys_result)
    sys_eval = judge.evaluate(sys_result["overall_summary"], trajectory_text=traj)
    _print_eval(sys_eval)
    results["sysllm"] = {**sys_result, "eval": sys_eval}

    # --- 3. MCTS ---
    print("\n" + "=" * 60)
    print(f"[3/4] MCTS-XRL: Step {args.step} を反復改善中 (iterations={args.iterations})...")
    print("=" * 60)
    mcts_result = MCTSXRL(llm, loader, iterations=args.iterations).explain_mcts(args.step)
    _print_mcts_result(mcts_result)
    mcts_eval = judge.evaluate(mcts_result["explanation"], ctx)
    _print_eval(mcts_eval)
    results["mcts"] = {**mcts_result, "eval": mcts_eval}

    # --- 4. TalkToAgent ---
    print("\n" + "=" * 60)
    print(f"[4/4] TalkToAgent: 質問「{query}」")
    print("=" * 60)
    tta_result = TalkToAgent(llm, loader).answer(query)
    _print_talktoagent_result(tta_result)
    tta_eval = judge.evaluate(tta_result["explanation"], ctx)
    _print_eval(tta_eval)
    results["talktoagent"] = {**tta_result, "eval": tta_eval}

    # --- 比較テーブル ---
    _print_compare_table(args.step, results)

    # --- JSON 保存 ---
    if args.output:
        _save_results(args.output, args.step, results)


def _print_compare_table(step: int, results: dict) -> None:
    """3手法の評価スコアを横並びで表示する。"""
    print("\n" + "=" * 60)
    print(f"  比較結果サマリー (Step {step})")
    print("=" * 60)

    header = f"{'手法':<12} {'Soundness':>10} {'Fidelity':>10} {'合計':>6}"
    print(header)
    print("-" * 42)

    method_labels = {"cot": "CoT", "sysllm": "SySLLM", "mcts": "MCTS-XRL", "talktoagent": "TalkToAgent"}
    for key, label in method_labels.items():
        ev = results[key]["eval"]
        s  = ev.get("soundness", "?")
        f  = ev.get("fidelity",  "?")
        t  = (s if isinstance(s, int) else 0) + (f if isinstance(f, int) else 0)
        print(f"{label:<12} {str(s)+'/2':>10} {str(f)+'/2':>10} {str(t)+'/4':>6}")

    print("-" * 42)
    # 最高スコアの手法を表示
    scored = {
        k: (results[k]["eval"].get("soundness", 0) or 0)
         + (results[k]["eval"].get("fidelity",  0) or 0)
        for k in results
    }
    best = max(scored, key=scored.get)
    print(f"  最高スコア: {method_labels[best]} ({scored[best]}/4)")
    print("=" * 60)


def _save_results(output_path: str, step: int, results: dict) -> None:
    """比較結果を JSON ファイルに保存する。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # context の dict は JSON シリアライズ可能だが numpy float は不可なので変換
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        try:
            return float(obj)
        except (TypeError, ValueError):
            return obj

    payload = {"step": step, "results": _clean(results)}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[保存完了] {out}")


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
    for i, node in enumerate(result["tree_summary"]):
        print(
            f"  ノード{i}: Q={node['q_value']:.3f}, "
            f"訪問={node['visits']}, "
            f"説明={node['explanation_snippet']}"
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
    s = eval_result.get("soundness", "?")
    f = eval_result.get("fidelity",  "?")
    t = (s if isinstance(s, int) else 0) + (f if isinstance(f, int) else 0)
    print(f"\n  Soundness={s}/2  Fidelity={f}/2  合計={t}/4")
    print(f"  理由: {eval_result.get('reason', '')}")


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
