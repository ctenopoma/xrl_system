"""baseline 評価スクリプト - 学習なしで Soundness/Fidelity を測定する。

設定の組み合わせ (テンプレート × プロンプト戦略 × 事前情報) を指定して
複数ステップを評価し、設定メタデータ付きの JSON を保存する。

使い方:
    # デフォルト設定 (v1_basic × zero_shot) で step 50,100,150 を評価
    python evaluate_baseline.py --steps 50 100 150

    # テンプレートとプロンプト戦略を指定
    python evaluate_baseline.py --steps 50 100 150 \\
        --template v1_combat_only --strategy cot

    # 事前情報あり (SySLLM の要約を注入)
    python evaluate_baseline.py --steps 50 100 150 \\
        --template v2_with_prior --prior-info sysllm

    # 全ステップを 20 件サンプリングして評価
    python evaluate_baseline.py --sample 20

    # 複数テンプレートを一括で比較
    python evaluate_baseline.py --steps 100 150 \\
        --template v1_basic v1_combat_only v2_with_prior

環境変数:
    XRL_MODEL_NAME  使用する LLM モデル名 (デフォルト: gpt-4o)
    XRL_API_KEY     API キー
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modules.data_loader import DataLoader
from modules.evaluator import EpisodeEvaluator
from modules.inference_engine import InferenceEngine, PromptingStrategy
from modules.llm_client import LLMClient
from modules.prompt_template import PromptTemplate, PRESETS
from modules.sysllm import SySLLM


# ------------------------------------------------------------------
# CLI パーサー
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="学習なし baseline 評価",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--steps", nargs="+", type=int, default=None,
        help="評価するステップ番号 (省略時は --sample で自動選択)",
    )
    p.add_argument(
        "--sample", type=int, default=10,
        help="--steps 未指定時にランダムサンプリングするステップ数 (デフォルト: 10)",
    )
    p.add_argument(
        "--template", nargs="+",
        choices=list(PRESETS.keys()), default=["v1_basic"],
        help="使用するプロンプトテンプレート (複数指定で一括比較)",
    )
    p.add_argument(
        "--strategy", choices=["zero_shot", "cot"], default="zero_shot",
        help="プロンプト戦略 (デフォルト: zero_shot)",
    )
    p.add_argument(
        "--prior-info", choices=["none", "sysllm"], default="none",
        dest="prior_info",
        help="事前情報の注入方法 (v2_with_prior テンプレート時のみ有効)",
    )
    p.add_argument(
        "--csv", default="data/trajectory_log.csv",
        help="軌跡ログ CSV (デフォルト: data/trajectory_log.csv)",
    )
    p.add_argument(
        "--model", default=None,
        help="LLM モデル名 (省略時は XRL_MODEL_NAME 環境変数または gpt-4o)",
    )
    p.add_argument(
        "--output-dir", default="results/baseline",
        dest="output_dir",
        help="結果 JSON の保存先ディレクトリ (デフォルト: results/baseline)",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="サンプリング乱数シード (デフォルト: 0)",
    )
    return p


# ------------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    llm    = LLMClient(model=args.model)
    loader = DataLoader(csv_path=args.csv)
    loader.load()
    judge  = EpisodeEvaluator(llm)

    # 評価ステップを決定
    steps = _resolve_steps(args, loader)
    print(f"評価ステップ: {steps}")

    # SySLLM 事前情報を先に生成 (必要な場合のみ)
    sysllm_summary = ""
    if args.prior_info == "sysllm":
        print("\n[SySLLM] エピソード要約を生成中...")
        sys_result = SySLLM(llm, loader).analyze()
        sysllm_summary = sys_result.get("overall_summary", "")
        print(f"  → {sysllm_summary[:100]}...")

    # テンプレートごとに評価
    all_results: list[dict] = []

    for template_id in args.template:
        print(f"\n{'=' * 60}")
        print(f"テンプレート: {template_id} | 戦略: {args.strategy} | 事前情報: {args.prior_info}")
        print("=" * 60)

        tpl    = PromptTemplate.from_preset(template_id)
        engine = InferenceEngine(
            llm=llm,
            template=tpl,
            strategy=PromptingStrategy(args.strategy),
        )

        step_results: list[dict] = []

        for step in steps:
            print(f"\n  [Step {step}] 説明生成中...", end=" ", flush=True)
            try:
                context = loader.get_step_context(step)

                # 事前情報の解決
                prior_info = ""
                if args.prior_info == "sysllm" and tpl.config.prior_info_slot_enabled:
                    prior_info = sysllm_summary

                # 説明生成
                explanation = engine.generate(context, prior_info=prior_info)

                # 評価
                eval_result = judge.evaluate(explanation, context=context)
                total = eval_result.get("soundness", 0) + eval_result.get("fidelity", 0)
                print(f"Soundness={eval_result['soundness']}/2 Fidelity={eval_result['fidelity']}/2 合計={total}/4")

                step_results.append({
                    "step":        step,
                    "explanation": explanation,
                    "eval":        eval_result,
                })
            except Exception as e:
                print(f"エラー: {e}")
                step_results.append({
                    "step":  step,
                    "error": str(e),
                    "eval":  {"soundness": 0, "fidelity": 0, "reason": str(e)},
                })

        # サマリー集計
        summary = _summarize(step_results)
        _print_summary(template_id, args.strategy, summary)

        all_results.append({
            "config": {
                **engine.to_dict(),
                "prior_info_mode": args.prior_info,
            },
            "steps":   step_results,
            "summary": summary,
        })

    # JSON 保存
    _save(args, steps, all_results)


# ------------------------------------------------------------------
# ヘルパー
# ------------------------------------------------------------------

def _resolve_steps(args: argparse.Namespace, loader: DataLoader) -> list[int]:
    """評価ステップリストを決定する。"""
    if args.steps:
        return sorted(args.steps)
    df = loader.load()
    available = df["step"].unique()
    rng = np.random.default_rng(args.seed)
    sampled = sorted(rng.choice(available, size=min(args.sample, len(available)), replace=False).tolist())
    return [int(s) for s in sampled]


def _summarize(step_results: list[dict]) -> dict:
    """ステップ結果リストから平均スコアを計算する。"""
    valid = [r for r in step_results if "error" not in r]
    if not valid:
        return {"n_steps": 0, "soundness_mean": 0.0, "fidelity_mean": 0.0, "total_mean": 0.0}
    soundness_vals = [r["eval"]["soundness"] for r in valid]
    fidelity_vals  = [r["eval"]["fidelity"]  for r in valid]
    return {
        "n_steps":        len(valid),
        "soundness_mean": round(float(np.mean(soundness_vals)), 3),
        "soundness_std":  round(float(np.std(soundness_vals)),  3),
        "fidelity_mean":  round(float(np.mean(fidelity_vals)),  3),
        "fidelity_std":   round(float(np.std(fidelity_vals)),   3),
        "total_mean":     round(float(np.mean([s + f for s, f in zip(soundness_vals, fidelity_vals)])), 3),
    }


def _print_summary(template_id: str, strategy: str, summary: dict) -> None:
    print(f"\n  --- サマリー ({template_id} / {strategy}) ---")
    print(f"  評価ステップ数: {summary['n_steps']}")
    print(f"  Soundness 平均: {summary['soundness_mean']:.2f} ± {summary.get('soundness_std', 0):.2f}")
    print(f"  Fidelity  平均: {summary['fidelity_mean']:.2f} ± {summary.get('fidelity_std', 0):.2f}")
    print(f"  合計スコア平均: {summary['total_mean']:.2f} / 4.00")


def _save(args: argparse.Namespace, steps: list[int], all_results: list[dict]) -> None:
    """結果を JSON ファイルに保存する。"""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    templates_str = "_".join(args.template)
    filename = f"baseline_{templates_str}_{args.strategy}_{ts}.json"
    out_path = out_dir / filename

    payload = {
        "run_at":    ts,
        "model":     args.model or "env:XRL_MODEL_NAME",
        "steps":     steps,
        "prior_info_mode": args.prior_info,
        "results":   all_results,
    }

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        try:
            return float(obj)
        except (TypeError, ValueError):
            return obj

    out_path.write_text(
        json.dumps(_clean(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[保存完了] {out_path}")


# ------------------------------------------------------------------
# エントリーポイント
# ------------------------------------------------------------------

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run(build_parser().parse_args())
