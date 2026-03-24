"""スイープ実験ランナー。

YAML 設定ファイルを読み込み、
  build_dataset → train_lora → evaluate_baseline (local)
のパイプラインを全組み合わせで実行する。

進捗は experiments/<exp_id>/progress.json に記録され、
中断後に再実行すると完了済みの combo をスキップして再開できる。

使い方:
    # 全組み合わせを実行
    python run_experiment.py --config configs/sweep_example.yaml

    # 実行前に組み合わせ一覧を表示して確認 (何も実行しない)
    python run_experiment.py --config configs/sweep_example.yaml --dry-run

    # baseline 評価 (外部 API) もまとめて実行
    python run_experiment.py --config configs/sweep_example.yaml --eval-baseline

    # 途中から再開 (progress.json が残っている場合)
    python run_experiment.py --config configs/sweep_example.yaml \
        --resume experiments/sweep_20260324_001
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

EXPERIMENTS_DIR = Path("experiments")

# スイープ可能なパラメータキー
_SWEEP_KEYS = [
    "template",
    "label_source",
    "min_score",
    "lora_rank",
    "num_epochs",
    "mcts_iterations",
    "lora_alpha",
    "batch_size",
    "grad_accum",
    "max_length",
]


# ------------------------------------------------------------------
# CLI パーサー
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="スイープ実験ランナー (build_dataset → train_lora → evaluate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        required=True,
        help="スイープ設定 YAML ファイルのパス",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="combo 一覧を表示するだけで実際には実行しない",
    )
    p.add_argument(
        "--eval-baseline",
        action="store_true",
        dest="eval_baseline",
        help="スイープ開始前に外部 API で baseline 評価を実行する",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="既存の実験ディレクトリを指定して途中から再開",
    )
    p.add_argument(
        "--output-dir",
        default=str(EXPERIMENTS_DIR),
        dest="output_dir",
        help=f"実験ディレクトリの保存先 (デフォルト: {EXPERIMENTS_DIR})",
    )
    return p


# ------------------------------------------------------------------
# メイン
# ------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    config = _load_yaml(args.config)

    combos = _build_combos(config)
    print(f"[Experiment] 設定ファイル: {args.config}")
    print(f"[Experiment] 合計 combo 数: {len(combos)}")

    if args.dry_run:
        _print_combos(combos, config)
        return

    # 実験ディレクトリの準備
    if args.resume:
        exp_dir = Path(args.resume)
        if not exp_dir.exists():
            print(f"[エラー] 再開対象のディレクトリが見つかりません: {exp_dir}")
            sys.exit(1)
        progress = _load_progress(exp_dir)
        print(f"[Experiment] 再開: {exp_dir}")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"sweep_{ts}"
        exp_dir = Path(args.output_dir) / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        progress = _init_progress(exp_id, combos, config)
        _save_progress(exp_dir, progress)
        print(f"[Experiment] 実験 ID: {exp_id}")
        print(f"[Experiment] 保存先: {exp_dir}")

    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # --eval-baseline: 外部 API baseline 評価を最初に実行
    if args.eval_baseline:
        print("\n" + "=" * 60)
        print("  [Phase 0] Baseline 評価 (外部 API)")
        print("=" * 60)
        _run_baseline_eval(config, log_dir)

    # 各 combo を実行
    for combo_idx, combo_info in enumerate(progress["combos"]):
        if combo_info["status"] == "complete":
            print(f"\n[combo {combo_idx + 1}/{len(combos)}] スキップ (完了済み): {combo_info['config']}")
            continue

        combo = combo_info["config"]
        print(f"\n{'=' * 60}")
        print(f"[combo {combo_idx + 1}/{len(combos)}] {combo}")
        print("=" * 60)

        combo_log_dir = log_dir / f"combo_{combo_idx:03d}"
        combo_log_dir.mkdir(exist_ok=True)

        try:
            # Phase 1: データセット生成
            dataset_id = _run_build_dataset(combo, config, combo_log_dir)
            if dataset_id is None:
                combo_info["status"] = "failed"
                combo_info["error"] = "build_dataset failed"
                _save_progress(exp_dir, progress)
                continue

            # Phase 2: LoRA 学習
            run_id = _run_train_lora(combo, config, dataset_id, combo_log_dir)
            if run_id is None:
                combo_info["status"] = "failed"
                combo_info["error"] = "train_lora failed"
                _save_progress(exp_dir, progress)
                continue

            # Phase 3: 学習後評価
            adapter_dir = Path("models") / run_id / "adapter"
            ok = _run_eval_local(combo, config, adapter_dir, combo_log_dir)

            combo_info["status"] = "complete" if ok else "eval_failed"
            combo_info["dataset_id"] = dataset_id
            combo_info["run_id"] = run_id
            combo_info["adapter"] = str(adapter_dir)

        except Exception as e:
            combo_info["status"] = "failed"
            combo_info["error"] = str(e)
            print(f"[エラー] combo {combo_idx} で例外発生: {e}")

        _save_progress(exp_dir, progress)

    # 完了サマリー
    n_ok = sum(1 for c in progress["combos"] if c["status"] == "complete")
    n_fail = sum(1 for c in progress["combos"] if c["status"] in ("failed", "eval_failed"))
    print(f"\n{'=' * 60}")
    print(f"  実験完了: {n_ok}/{len(combos)} 成功 | {n_fail} 失敗")
    print(f"  結果比較: python compare_results.py --summary results/baseline/baseline_summary.csv")
    print("=" * 60)


# ------------------------------------------------------------------
# フェーズ実行
# ------------------------------------------------------------------

def _run_baseline_eval(config: dict, log_dir: Path) -> bool:
    """外部 API で baseline 評価を実行する。"""
    cmd = [sys.executable, "evaluate_baseline.py"]
    _add_eval_args(cmd, config, backend="external")
    return _exec(cmd, log_dir / "baseline_eval.log") == 0


def _run_build_dataset(
    combo: dict, config: dict, log_dir: Path
) -> str | None:
    """build_dataset.py を実行し、生成された dataset_id を返す。"""
    cmd = [
        sys.executable, "build_dataset.py",
        "--label-source",    str(combo.get("label_source", "cot")),
        "--template",        str(combo.get("template", "v1_basic")),
        "--min-score",       str(combo.get("min_score", 0.0)),
        "--mcts-iterations", str(combo.get("mcts_iterations", 4)),
        "--csv",             str(config.get("csv", "data/trajectory_log.csv")),
    ]
    log_path = log_dir / "build_dataset.log"
    rc = _exec(cmd, log_path)
    if rc != 0:
        return None

    # ログから dataset_id を抽出
    return _extract_value_from_log(log_path, "dataset_id :")


def _run_train_lora(
    combo: dict, config: dict, dataset_id: str, log_dir: Path
) -> str | None:
    """train_lora.py を実行し、run_id を返す。"""
    from modules.dataset_builder import DATASETS_DIR

    dataset_dir = DATASETS_DIR / dataset_id
    alpha = combo.get("lora_alpha") or (combo.get("lora_rank", 16) * 2)

    cmd = [
        sys.executable, "train_lora.py",
        "--dataset",    str(dataset_dir),
        "--base-model", str(config.get("base_model", "Qwen/Qwen2.5-7B-Instruct")),
        "--rank",       str(combo.get("lora_rank", 16)),
        "--alpha",      str(alpha),
        "--epochs",     str(combo.get("num_epochs", 3)),
        "--batch-size", str(combo.get("batch_size", 4)),
        "--grad-accum", str(combo.get("grad_accum", 4)),
        "--max-length", str(combo.get("max_length", 1024)),
    ]
    log_path = log_dir / "train_lora.log"
    rc = _exec(cmd, log_path)
    if rc != 0:
        return None

    # ログから run_id を抽出
    return _extract_value_from_log(log_path, "run_id   :")


def _run_eval_local(
    combo: dict, config: dict, adapter_dir: Path, log_dir: Path
) -> bool:
    """学習済みアダプタで evaluate_baseline.py を実行する。"""
    cmd = [
        sys.executable, "evaluate_baseline.py",
        "--backend", "local",
        "--adapter", str(adapter_dir),
        "--template", str(combo.get("template", "v1_basic")),
    ]
    _add_eval_args(cmd, config, backend="local")
    return _exec(cmd, log_dir / "eval_local.log") == 0


def _add_eval_args(cmd: list, config: dict, backend: str = "external") -> None:
    """evaluate_baseline.py の共通引数を追加する。"""
    eval_steps = config.get("eval_steps")
    eval_sample = config.get("eval_sample")
    strategy = config.get("eval_strategy", "zero_shot")

    if eval_steps:
        cmd += ["--steps"] + [str(s) for s in eval_steps]
    elif eval_sample:
        cmd += ["--sample", str(eval_sample)]

    cmd += ["--strategy", strategy]
    cmd += ["--csv", str(config.get("csv", "data/trajectory_log.csv"))]


# ------------------------------------------------------------------
# 補助関数
# ------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_combos(config: dict) -> list[dict]:
    """設定からスイープ軸を取り出し、全組み合わせの dict リストを返す。"""
    axes: dict[str, list] = {}
    for key in _SWEEP_KEYS:
        val = config.get(key)
        if val is None:
            continue
        axes[key] = val if isinstance(val, list) else [val]

    if not axes:
        return [{}]

    keys = list(axes.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes.values())]


def _print_combos(combos: list[dict], config: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Dry-run: {len(combos)} 件の combo")
    print("=" * 60)
    for i, c in enumerate(combos):
        print(f"  [{i + 1:3d}] {c}")
    print(f"\n  固定パラメータ:")
    for key in ("base_model", "num_epochs", "batch_size", "eval_steps", "eval_strategy"):
        val = config.get(key)
        if val is not None:
            print(f"    {key}: {val}")


def _init_progress(exp_id: str, combos: list[dict], config: dict) -> dict:
    return {
        "experiment_id": exp_id,
        "started_at": datetime.now().isoformat(),
        "config_file": str(Path(config.get("_source", "unknown"))),
        "combos": [
            {"id": f"combo_{i:03d}", "config": c, "status": "pending"}
            for i, c in enumerate(combos)
        ],
    }


def _load_progress(exp_dir: Path) -> dict:
    path = exp_dir / "progress.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_progress(exp_dir: Path, progress: dict) -> None:
    progress["updated_at"] = datetime.now().isoformat()
    with open(exp_dir / "progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def _exec(cmd: list, log_path: Path) -> int:
    """コマンドを実行し、出力をログファイルに書きつつ、リターンコードを返す。"""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"  $ {cmd_str}")
    print(f"  → ログ: {log_path}")

    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"$ {cmd_str}\n\n")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        f.write(result.stdout)

    # ログの末尾数行を stdout に表示
    lines = result.stdout.strip().splitlines()
    for line in lines[-6:]:
        print(f"    {line}")

    if result.returncode != 0:
        print(f"  [失敗] 終了コード: {result.returncode}")
    return result.returncode


def _extract_value_from_log(log_path: Path, marker: str) -> str | None:
    """ログファイルから "marker <value>" パターンを抽出する。"""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if marker in line:
                parts = line.split(marker, 1)
                if len(parts) == 2:
                    return parts[1].strip()
    except Exception:
        pass
    return None


# ------------------------------------------------------------------
# エントリーポイント
# ------------------------------------------------------------------

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()
    # YAML に _source を付けてデバッグ用に残す
    cfg = _load_yaml(args.config)
    cfg["_source"] = args.config
    run(args)
