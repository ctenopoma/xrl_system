"""baseline 評価スクリプト - 学習なし/学習後の Soundness/Fidelity を測定する。

設定の組み合わせ (テンプレート × プロンプト戦略 × 事前情報) を指定して
複数ステップを評価し、設定メタデータ付きの CSV を保存する。
学習前後で同じコマンドを使うことで baseline_summary.csv に比較データが蓄積される。

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

    # 学習後のローカル LoRA モデルで評価 (Judge は外部 API を使用)
    python evaluate_baseline.py --sample 20 \\
        --backend local --adapter models/run_xxx/adapter

    # ベースモデルを明示指定 (run_config.json がない場合)
    python evaluate_baseline.py --sample 20 \\
        --backend local --adapter models/run_xxx/adapter \\
        --base-model Qwen/Qwen2.5-7B-Instruct

環境変数:
    XRL_MODEL_NAME  Judge (評価) に使用する LLM モデル名 (デフォルト: gpt-4o)
    XRL_API_KEY     API キー (--backend local でも Judge 用に必要)
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
from modules.inference_engine import InferenceEngine, LocalLoRABackend, PromptingStrategy
from modules.llm_client import LLMClient
from modules.mcts_xrl import MCTSXRL
from modules.prompt_template import PromptTemplate, PRESETS
from modules.sysllm import SySLLM
from modules.talktoagent import TalkToAgent

METHOD_CHOICES = ["zero_shot", "cot", "mcts", "sysllm", "agent"]


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
        "--method", nargs="+",
        choices=METHOD_CHOICES, default=None,
        help=(
            "比較する XRL 手法 (複数指定で一括比較)。"
            " 指定すると --template/--strategy を上書きして手法軸で比較する。"
            f" 選択肢: {METHOD_CHOICES}"
        ),
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

    # --- ローカル LoRA バックエンド ---
    local_grp = p.add_argument_group("ローカル LoRA オプション (--backend local 時に有効)")
    local_grp.add_argument(
        "--backend",
        choices=["external", "local"],
        default="external",
        help="推論バックエンド: external=外部API, local=LoRAアダプタ (デフォルト: external)",
    )
    local_grp.add_argument(
        "--adapter",
        default=None,
        help="LoRA アダプタのパス (--backend local 必須)",
    )
    local_grp.add_argument(
        "--base-model",
        default=None,
        dest="base_model",
        help="ベースモデルの HuggingFace ID (省略時は adapter/../run_config.json から読み込み)",
    )
    local_grp.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        dest="max_new_tokens",
        help="ローカル推論の最大生成トークン数 (デフォルト: 512)",
    )

    return p


# ------------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # バリデーション
    if args.backend == "local" and not args.adapter:
        print("[エラー] --backend local を使う場合は --adapter が必要です。")
        import sys
        sys.exit(1)

    loader = DataLoader(csv_path=args.csv)
    loader.load()

    # --- 生成バックエンド ---
    if args.backend == "local":
        gen_backend = LocalLoRABackend(
            adapter_path=args.adapter,
            base_model=args.base_model,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"[backend] ローカル LoRA: {args.adapter}")
    else:
        gen_backend = LLMClient(model=args.model)
        print(f"[backend] 外部 API: {gen_backend.model}")

    # --- Judge は常に外部 API を使用 ---
    judge_llm = LLMClient(model=args.model)
    judge = EpisodeEvaluator(judge_llm)

    # 評価ステップを決定
    steps = _resolve_steps(args, loader)
    print(f"評価ステップ: {steps}")

    # ------------------------------------------------------------------
    # --method モード: XRL 手法を横断比較
    # ------------------------------------------------------------------
    if args.method:
        all_results = _run_method_comparison(args, loader, gen_backend, judge, steps)
        _save(args, steps, all_results)
        return

    # ------------------------------------------------------------------
    # 従来モード: --template × --strategy
    # ------------------------------------------------------------------

    # SySLLM 事前情報を先に生成 (必要な場合のみ)
    sysllm_summary = ""
    if args.prior_info == "sysllm":
        print("\n[SySLLM] エピソード要約を生成中...")
        sys_result = SySLLM(judge_llm, loader).analyze()
        sysllm_summary = sys_result.get("overall_summary", "")
        print(f"  → {sysllm_summary[:100]}...")

    # テンプレートごとに評価
    all_results: list[dict] = []

    for template_id in args.template:
        print(f"\n{'=' * 60}")
        print(
            f"テンプレート: {template_id} | 戦略: {args.strategy}"
            f" | 事前情報: {args.prior_info} | backend: {args.backend}"
        )
        print("=" * 60)

        tpl    = PromptTemplate.from_preset(template_id)
        engine = InferenceEngine(
            llm=gen_backend,
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
            "method": f"{template_id}/{args.strategy}",
            "config": {
                **engine.to_dict(),
                "prior_info_mode": args.prior_info,
                "backend":         args.backend,
                "adapter":         args.adapter or "",
            },
            "steps":   step_results,
            "summary": summary,
        })

    # CSV / JSON 保存
    _save(args, steps, all_results)


# ------------------------------------------------------------------
# ヘルパー
# ------------------------------------------------------------------

def _generate_for_method(
    method: str,
    step: int,
    context,
    engines: dict,
    mcts_xrl: "MCTSXRL",
    talk_agent: "TalkToAgent",
    sysllm_summary: str,
) -> str:
    """指定手法でステップの説明文を生成して返す。"""
    if method == "zero_shot":
        return engines["zero_shot"].generate(context)
    if method == "cot":
        return engines["cot"].generate(context)
    if method == "mcts":
        return mcts_xrl.explain_mcts(step)["explanation"]
    if method == "sysllm":
        return engines["sysllm"].generate(context, prior_info=sysllm_summary)
    if method == "agent":
        query = f"Step {step} でエージェントがなぜその操舵をしたのか詳しく説明してください。"
        return talk_agent.answer(query)["explanation"]
    raise ValueError(f"未知の手法: {method}")


def _run_method_comparison(
    args: argparse.Namespace,
    loader: DataLoader,
    gen_backend,
    judge: "EpisodeEvaluator",
    steps: list[int],
) -> list[dict]:
    """--method モード: 全指定手法を同一ステップで評価して結果リストを返す。"""
    # エンジン群の初期化
    engines = {
        "zero_shot": InferenceEngine(
            gen_backend, PromptTemplate.from_preset("v1_basic"),
            strategy=PromptingStrategy.ZERO_SHOT,
        ),
        "cot": InferenceEngine(
            gen_backend, PromptTemplate.from_preset("v1_basic"),
            strategy=PromptingStrategy.COT,
        ),
        "sysllm": InferenceEngine(
            gen_backend, PromptTemplate.from_preset("v2_with_prior"),
            strategy=PromptingStrategy.ZERO_SHOT,
        ),
    }
    mcts_xrl   = MCTSXRL(gen_backend, loader)
    talk_agent = TalkToAgent(gen_backend, loader)

    # sysllm 手法が含まれる場合のみエピソード要約を事前生成
    sysllm_summary = ""
    if "sysllm" in args.method:
        print("\n[SySLLM] エピソード要約を生成中...")
        sys_result    = SySLLM(gen_backend, loader).analyze()
        sysllm_summary = sys_result.get("overall_summary", "")
        print(f"  → {sysllm_summary[:100]}...")

    all_results: list[dict] = []

    for method in args.method:
        print(f"\n{'=' * 60}")
        print(f"手法: {method}")
        print("=" * 60)

        step_results: list[dict] = []
        for step in steps:
            print(f"\n  [Step {step}] {method} 生成中...", end=" ", flush=True)
            try:
                context     = loader.get_step_context(step)
                explanation = _generate_for_method(
                    method, step, context,
                    engines, mcts_xrl, talk_agent, sysllm_summary,
                )
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

        summary = _summarize(step_results)
        _print_summary(method, "-", summary)

        all_results.append({
            "method":  method,
            "config":  {
                "template": {"template_id": method},
                "strategy": method,
                "prior_info_mode": "sysllm" if method == "sysllm" else "none",
                "backend": args.backend,
                "adapter": args.adapter or "",
            },
            "steps":   step_results,
            "summary": summary,
        })

    return all_results


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
    """結果を JSON / CSV ファイルに保存する。

    出力ファイル:
        baseline_steps_<ts>.csv   : ステップ単位の詳細 (1行 = 1設定 × 1ステップ)
        baseline_summary.csv      : 設定ごとのサマリー (実行ごとに追記)
    """
    import csv

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # model 識別子: ローカル時はアダプタパス、外部時はモデル名
    model = (
        f"local:{args.adapter}"
        if args.backend == "local" and args.adapter
        else (args.model or "env:XRL_MODEL_NAME")
    )

    # ------------------------------------------------------------------
    # 1. ステップ単位 CSV (毎回新規ファイル)
    # ------------------------------------------------------------------
    steps_csv = out_dir / f"baseline_steps_{ts}.csv"
    step_fields = [
        "run_at", "model", "backend", "method", "template_id", "strategy", "prior_info_mode",
        "step", "soundness", "fidelity", "total", "explanation", "reason",
    ]
    with steps_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=step_fields)
        writer.writeheader()
        for run in all_results:
            cfg = run["config"]
            for sr in run["steps"]:
                ev = sr.get("eval", {})
                s  = ev.get("soundness", 0)
                fi = ev.get("fidelity",  0)
                writer.writerow({
                    "run_at":          ts,
                    "model":           model,
                    "backend":         cfg.get("backend", "external"),
                    "method":          run.get("method", ""),
                    "template_id":     cfg["template"]["template_id"],
                    "strategy":        cfg["strategy"],
                    "prior_info_mode": cfg["prior_info_mode"],
                    "step":            sr["step"],
                    "soundness":       s,
                    "fidelity":        fi,
                    "total":           s + fi,
                    "explanation":     sr.get("explanation", ""),
                    "reason":          ev.get("reason", ""),
                })
    print(f"\n[保存完了] {steps_csv}  (ステップ詳細)")

    # ------------------------------------------------------------------
    # 2. サマリー CSV (実行ごとに追記)
    # ------------------------------------------------------------------
    summary_csv = out_dir / "baseline_summary.csv"
    summary_fields = [
        "run_at", "model", "backend", "method", "template_id", "strategy", "prior_info_mode",
        "n_steps",
        "soundness_mean", "soundness_std",
        "fidelity_mean",  "fidelity_std",
        "total_mean",
    ]
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        if write_header:
            writer.writeheader()
        for run in all_results:
            cfg = run["config"]
            sm  = run["summary"]
            writer.writerow({
                "run_at":          ts,
                "model":           model,
                "backend":         cfg.get("backend", "external"),
                "method":          run.get("method", ""),
                "template_id":     cfg["template"]["template_id"],
                "strategy":        cfg["strategy"],
                "prior_info_mode": cfg["prior_info_mode"],
                "n_steps":         sm["n_steps"],
                "soundness_mean":  sm["soundness_mean"],
                "soundness_std":   sm.get("soundness_std", ""),
                "fidelity_mean":   sm["fidelity_mean"],
                "fidelity_std":    sm.get("fidelity_std", ""),
                "total_mean":      sm["total_mean"],
            })
    print(f"[保存完了] {summary_csv}  (サマリー・追記)")


# ------------------------------------------------------------------
# エントリーポイント
# ------------------------------------------------------------------

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run(build_parser().parse_args())
