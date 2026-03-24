"""評価結果の比較レポートを生成する。

baseline_summary.csv を読み込み、
  外部 API (external) の結果をベースラインとして、
  LoRA 学習後 (local) の結果との差分をマークダウン表で出力する。

使い方:
    python compare_results.py \
        --summary results/baseline/baseline_summary.csv \
        --output  results/comparison_report.md

    # stdout に出力するだけ (ファイル保存しない)
    python compare_results.py \
        --summary results/baseline/baseline_summary.csv

    # ベースラインのみ表示 (local 行がなくてもエラーにならない)
    python compare_results.py \
        --summary results/baseline/baseline_summary.csv \
        --show-baseline-only
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import NamedTuple


# ------------------------------------------------------------------
# データ型
# ------------------------------------------------------------------

class Row(NamedTuple):
    run_at: str
    model: str
    backend: str          # "external" | "local"
    template_id: str
    strategy: str
    prior_info_mode: str
    n_steps: int
    soundness_mean: float
    soundness_std: float
    fidelity_mean: float
    fidelity_std: float
    total_mean: float


def _parse_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _parse_int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


# ------------------------------------------------------------------
# CSV 読み込み
# ------------------------------------------------------------------

def load_summary(path: Path) -> list[Row]:
    rows: list[Row] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            rows.append(Row(
                run_at          = rec.get("run_at", ""),
                model           = rec.get("model", ""),
                backend         = rec.get("backend", "external"),
                template_id     = rec.get("template_id", ""),
                strategy        = rec.get("strategy", ""),
                prior_info_mode = rec.get("prior_info_mode", ""),
                n_steps         = _parse_int(rec.get("n_steps", "0")),
                soundness_mean  = _parse_float(rec.get("soundness_mean", "0")),
                soundness_std   = _parse_float(rec.get("soundness_std", "0")),
                fidelity_mean   = _parse_float(rec.get("fidelity_mean", "0")),
                fidelity_std    = _parse_float(rec.get("fidelity_std", "0")),
                total_mean      = _parse_float(rec.get("total_mean", "0")),
            ))
    return rows


# ------------------------------------------------------------------
# 比較ロジック
# ------------------------------------------------------------------

# グループキー: 同条件でベースラインを探す
_GROUP_KEYS = ("template_id", "strategy", "prior_info_mode")


def _group_key(r: Row) -> tuple:
    return (r.template_id, r.strategy, r.prior_info_mode)


def build_baseline_map(rows: list[Row]) -> dict[tuple, Row]:
    """グループキー → 最新の external 行 を返す。"""
    external = [r for r in rows if r.backend == "external"]
    # 同じキーで複数行ある場合は最新 (run_at 降順) を使う
    best: dict[tuple, Row] = {}
    for r in sorted(external, key=lambda x: x.run_at):
        best[_group_key(r)] = r
    return best


def compute_deltas(
    rows: list[Row],
    baseline_map: dict[tuple, Row],
) -> list[dict]:
    """各行にベースライン差分を付与した辞書リストを返す。"""
    results = []
    for r in rows:
        key = _group_key(r)
        bl = baseline_map.get(key)
        delta = (r.total_mean - bl.total_mean) if bl else None
        results.append({
            "row": r,
            "baseline": bl,
            "delta": delta,
        })
    return results


# ------------------------------------------------------------------
# マークダウン生成
# ------------------------------------------------------------------

def _fmt(val: float, decimals: int = 2) -> str:
    return f"{val:.{decimals}f}"


def _fmt_delta(delta: float | None) -> str:
    if delta is None:
        return "—"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}"


def generate_report(
    rows: list[Row],
    baseline_map: dict[tuple, Row],
    show_baseline_only: bool = False,
) -> str:
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append(f"# XRL System — 評価比較レポート")
    lines.append(f"")
    lines.append(f"生成日時: {now}")
    lines.append(f"")

    # ---- サマリーテーブル ----------------------------------------
    lines.append("## サマリー")
    lines.append("")

    header = [
        "backend", "template_id", "strategy", "prior_info_mode",
        "n_steps", "soundness", "fidelity", "total_mean", "vs baseline",
    ]
    rows_for_table: list[dict] = []

    # ベースライン行を先頭に追加
    for key, bl in sorted(baseline_map.items()):
        rows_for_table.append({
            "backend":         "(baseline)",
            "template_id":     bl.template_id,
            "strategy":        bl.strategy,
            "prior_info_mode": bl.prior_info_mode,
            "n_steps":         bl.n_steps,
            "soundness":       _fmt(bl.soundness_mean),
            "fidelity":        _fmt(bl.fidelity_mean),
            "total_mean":      _fmt(bl.total_mean),
            "vs baseline":     "—",
            "_sort":           bl.total_mean,
        })

    if not show_baseline_only:
        local_rows = [r for r in rows if r.backend == "local"]
        local_with_delta = compute_deltas(local_rows, baseline_map)
        local_with_delta.sort(key=lambda x: x["row"].total_mean, reverse=True)

        for item in local_with_delta:
            r = item["row"]
            rows_for_table.append({
                "backend":         "local",
                "template_id":     r.template_id,
                "strategy":        r.strategy,
                "prior_info_mode": r.prior_info_mode,
                "n_steps":         r.n_steps,
                "soundness":       _fmt(r.soundness_mean),
                "fidelity":        _fmt(r.fidelity_mean),
                "total_mean":      _fmt(r.total_mean),
                "vs baseline":     _fmt_delta(item["delta"]),
                "_sort":           r.total_mean,
            })

    # マークダウンテーブル描画
    col_keys = [
        "backend", "template_id", "strategy", "prior_info_mode",
        "n_steps", "soundness", "fidelity", "total_mean", "vs baseline",
    ]
    col_widths = {k: len(k) for k in col_keys}
    for row_d in rows_for_table:
        for k in col_keys:
            col_widths[k] = max(col_widths[k], len(str(row_d.get(k, ""))))

    def _row_str(vals: dict) -> str:
        cells = [str(vals.get(k, "")).ljust(col_widths[k]) for k in col_keys]
        return "| " + " | ".join(cells) + " |"

    def _sep_str() -> str:
        seps = ["-" * col_widths[k] for k in col_keys]
        return "| " + " | ".join(seps) + " |"

    lines.append(_row_str({k: k for k in col_keys}))
    lines.append(_sep_str())
    for row_d in rows_for_table:
        lines.append(_row_str(row_d))
    lines.append("")

    # ---- 統計サマリー -------------------------------------------
    if not show_baseline_only:
        local_rows = [r for r in rows if r.backend == "local"]
        if local_rows:
            best = max(local_rows, key=lambda r: r.total_mean)
            worst = min(local_rows, key=lambda r: r.total_mean)
            avg_total = sum(r.total_mean for r in local_rows) / len(local_rows)

            lines.append("## 統計")
            lines.append("")
            lines.append(f"- local run 数: **{len(local_rows)}**")
            lines.append(f"- total_mean 最高: **{_fmt(best.total_mean)}**"
                         f"  ({best.template_id} / {best.strategy} / {best.prior_info_mode})")
            lines.append(f"- total_mean 最低: **{_fmt(worst.total_mean)}**"
                         f"  ({worst.template_id} / {worst.strategy} / {worst.prior_info_mode})")
            lines.append(f"- total_mean 平均: **{_fmt(avg_total)}**")
            lines.append("")

    # ---- グループ別詳細 -----------------------------------------
    lines.append("## グループ別詳細")
    lines.append("")

    grouped: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows:
        grouped[_group_key(r)].append(r)

    for key in sorted(grouped.keys()):
        template_id, strategy, prior_info_mode = key
        lines.append(f"### `{template_id}` × `{strategy}` × `{prior_info_mode}`")
        lines.append("")

        group_rows = grouped[key]
        bl = baseline_map.get(key)

        ext_rows = [r for r in group_rows if r.backend == "external"]
        loc_rows = [r for r in group_rows if r.backend == "local"]

        if ext_rows:
            latest_ext = max(ext_rows, key=lambda r: r.run_at)
            lines.append(f"**Baseline (external)** — run_at: `{latest_ext.run_at}`  "
                         f"model: `{latest_ext.model}`")
            lines.append(f"- n_steps: {latest_ext.n_steps}")
            lines.append(f"- soundness: {_fmt(latest_ext.soundness_mean)} "
                         f"± {_fmt(latest_ext.soundness_std)}")
            lines.append(f"- fidelity:  {_fmt(latest_ext.fidelity_mean)} "
                         f"± {_fmt(latest_ext.fidelity_std)}")
            lines.append(f"- **total_mean: {_fmt(latest_ext.total_mean)}**")
            lines.append("")

        if loc_rows and not show_baseline_only:
            loc_rows_sorted = sorted(loc_rows, key=lambda r: r.total_mean, reverse=True)
            lines.append(f"**Local runs** ({len(loc_rows)} 件、total_mean 降順)")
            lines.append("")
            for i, r in enumerate(loc_rows_sorted, 1):
                delta = (r.total_mean - bl.total_mean) if bl else None
                lines.append(f"{i}. run_at `{r.run_at}` — "
                             f"soundness={_fmt(r.soundness_mean)}, "
                             f"fidelity={_fmt(r.fidelity_mean)}, "
                             f"**total={_fmt(r.total_mean)}** "
                             f"(vs baseline: {_fmt_delta(delta)})")
            lines.append("")
        elif not show_baseline_only:
            lines.append("*(local run なし)*")
            lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# エントリーポイント
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="baseline_summary.csv から比較レポートを生成する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--summary",
        required=True,
        help="baseline_summary.csv のパス",
    )
    p.add_argument(
        "--output",
        default=None,
        help="出力先 Markdown ファイルのパス (省略時は stdout)",
    )
    p.add_argument(
        "--show-baseline-only",
        action="store_true",
        dest="show_baseline_only",
        help="external 行のみ表示する (local 行がなくてもエラーにしない)",
    )
    return p


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"[エラー] ファイルが見つかりません: {summary_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_summary(summary_path)
    if not rows:
        print(f"[エラー] {summary_path} にデータがありません", file=sys.stderr)
        sys.exit(1)

    baseline_map = build_baseline_map(rows)
    report = generate_report(rows, baseline_map, args.show_baseline_only)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"[保存完了] {out_path}")
    else:
        print(report)

    # 簡易サマリーを stderr に表示
    n_ext = sum(1 for r in rows if r.backend == "external")
    n_loc = sum(1 for r in rows if r.backend == "local")
    print(f"\n[統計] external: {n_ext} 行 | local: {n_loc} 行 | 合計: {len(rows)} 行",
          file=sys.stderr)


if __name__ == "__main__":
    main()
