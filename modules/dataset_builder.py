"""DatasetBuilder — XRL手法でラベルを生成し、LoRA学習用 JSONL を作成する。

ラベルソース手法:
    cot    : MCTSXRL.explain_cot() で1回のLLM呼び出しで説明を生成
    mcts   : MCTSXRL.explain_mcts() で反復改善した高品質ラベルを生成
    sysllm : SySLLM でエピソードサマリーを生成し、各ステップの入力に prior_info
             として注入して CoT ラベルを生成 (v2_with_prior テンプレート推奨)

出力:
    datasets/<dataset_id>/
        train.jsonl   — 学習用 (80%)
        val.jsonl     — 検証用 (20%)
        metadata.json — 設定・統計の記録
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from modules.data_loader import DataLoader
from modules.evaluator import EpisodeEvaluator
from modules.llm_client import LLMClient
from modules.mcts_xrl import MCTSXRL
from modules.prompt_template import PromptTemplate
from modules.sysllm import SySLLM

DATASETS_DIR = Path("datasets")
_TRAIN_RATIO = 0.8


@dataclass
class LabelSourceConfig:
    """ラベル生成手法の設定。

    Attributes:
        method:          "cot" | "mcts" | "sysllm"
        min_score:       品質フィルタ。Soundness+Fidelity の合計がこの値未満の
                         サンプルを除外する (0.0 = フィルタなし, 最大 4.0)
        mcts_iterations: method="mcts" 時の MCTS イテレーション数
    """

    method: str = "cot"
    min_score: float = 0.0
    mcts_iterations: int = 4


class DatasetBuilder:
    """XRL手法でラベルを生成し、LoRA学習用の JSONL データセットを作成する。

    使い方::

        cfg = LabelSourceConfig(method="mcts", min_score=3.0)
        builder = DatasetBuilder(llm, loader, template=tpl, label_config=cfg)
        dataset_id = builder.build()
        # → datasets/<dataset_id>/ に保存

        # 特定ステップのみ
        dataset_id = builder.build(steps=[50, 100, 150])
    """

    def __init__(
        self,
        llm: LLMClient,
        loader: DataLoader,
        template: PromptTemplate,
        label_config: Optional[LabelSourceConfig] = None,
        evaluator: Optional[EpisodeEvaluator] = None,
        output_dir: str | Path = DATASETS_DIR,
        seed: int = 42,
    ) -> None:
        self.llm = llm
        self.loader = loader
        self.template = template
        self.label_config = label_config or LabelSourceConfig()
        self.evaluator = evaluator or EpisodeEvaluator(llm)
        self.output_dir = Path(output_dir)
        self.seed = seed

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def build(self, steps: Optional[Iterable[int]] = None) -> str:
        """データセットを生成して保存し、dataset_id を返す。

        Args:
            steps: 生成対象のステップ番号のイテラブル。
                   None の場合は CSV の全ステップを使用。

        Returns:
            dataset_id (str): 保存先ディレクトリ名
        """
        df = self.loader.load()
        if steps is None:
            step_list = sorted(df["step"].unique().tolist())
        else:
            step_list = sorted(set(int(s) for s in steps))

        cfg = self.label_config
        print(
            f"[DatasetBuilder] method={cfg.method} | template={self.template.config.template_id}"
            f" | min_score={cfg.min_score} | 対象ステップ数={len(step_list)}"
        )

        # sysllm: エピソードサマリーを先に1回生成
        prior_info = self._prepare_prior_info()

        # 各ステップのラベル生成
        mcts = MCTSXRL(self.llm, self.loader, iterations=cfg.mcts_iterations)
        samples: list[dict] = []
        n_skipped = 0

        for i, step in enumerate(step_list):
            print(
                f"  [Step {step}] ({i + 1}/{len(step_list)}) ... ",
                end="",
                flush=True,
            )
            sample = self._generate_sample(step, mcts, prior_info)
            if sample is None:
                n_skipped += 1
                print(f"スキップ (score < {cfg.min_score})")
            else:
                samples.append(sample)
                print(
                    f"score={sample['score']:.1f} "
                    f"(S={sample['soundness']} F={sample['fidelity']})"
                )

        print(
            f"\n[DatasetBuilder] 完了: {len(samples)} サンプル保存 / "
            f"{n_skipped} スキップ / {len(step_list)} 試行"
        )

        return self._save(samples, len(step_list))

    # ------------------------------------------------------------------
    # 非公開ヘルパー
    # ------------------------------------------------------------------

    def _prepare_prior_info(self) -> str:
        """sysllm method の場合、エピソードサマリーを生成して返す。"""
        if self.label_config.method != "sysllm":
            return ""

        print("[DatasetBuilder] SySLLM でエピソードサマリーを生成中...")
        sysllm = SySLLM(self.llm, self.loader)
        result = sysllm.analyze()
        summary = result.get("overall_summary", "")
        print(f"  → {len(summary)}字のサマリー生成完了")
        return summary

    def _generate_sample(
        self,
        step: int,
        mcts: MCTSXRL,
        prior_info: str,
    ) -> Optional[dict]:
        """1ステップ分のサンプルを生成し、スコアが min_score 未満なら None を返す。

        Returns:
            スコアが閾値以上の場合はサンプル dict、それ以外は None
        """
        try:
            context = self.loader.get_step_context(step)
        except ValueError as e:
            print(f"(ステップ取得失敗: {e})")
            return None

        # --- ラベル生成 ---
        try:
            label = self._generate_label(step, mcts, context, prior_info)
        except Exception as e:
            print(f"(ラベル生成失敗: {e})")
            return None

        # --- 品質評価 ---
        try:
            eval_result = self.evaluator.evaluate(label, context=context)
            score = float(eval_result.get("soundness", 0)) + float(eval_result.get("fidelity", 0))
        except Exception as e:
            print(f"(評価失敗: {e}) ", end="")
            eval_result = {"soundness": 0, "fidelity": 0, "reason": str(e)}
            score = 0.0

        # --- 品質フィルタ ---
        if score < self.label_config.min_score:
            return None

        # --- (instruction, input) の生成 ---
        instruction, input_text = self.template.format_step(context, prior_info)

        return {
            "step":         step,
            "instruction":  instruction,
            "input":        input_text,
            "output":       label,
            "score":        score,
            "soundness":    int(eval_result.get("soundness", 0)),
            "fidelity":     int(eval_result.get("fidelity", 0)),
            "eval_reason":  eval_result.get("reason", ""),
            "template_id":  self.template.config.template_id,
            "label_source": self.label_config.method,
        }

    def _generate_label(
        self,
        step: int,
        mcts: MCTSXRL,
        context: dict,
        prior_info: str,
    ) -> str:
        """ラベル文字列を生成する。"""
        method = self.label_config.method

        if method == "cot":
            result = mcts.explain_cot(step)
            return result["explanation"]

        if method == "mcts":
            result = mcts.explain_mcts(step)
            return result["explanation"]

        if method == "sysllm":
            # SySLLM サマリーを prior_info として CoT 生成
            # prior_info_slot_enabled=False のテンプレートでは prior_info は無視されるが
            # ラベル生成には使えるので別途プロンプトを構築する
            if self.template.config.prior_info_slot_enabled:
                # テンプレートが prior_info スロット対応なら format_step がそのまま使える
                system, user = self.template.format_step(context, prior_info)
            else:
                # prior_info スロットなし: system に要約を追記する
                system, user = self.template.format_step(context)
                if prior_info:
                    system += f"\n\n【エピソード全体の要約（参考情報）】\n{prior_info}"
            return self.llm.simple_prompt(system, user)

        raise ValueError(f"Unknown label method: {method!r}. Use 'cot', 'mcts', or 'sysllm'.")

    def _save(self, samples: list[dict], n_attempted: int) -> str:
        """サンプルを train/val 分割して保存し、dataset_id を返す。"""
        cfg = self.label_config
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        min_score_str = f"{cfg.min_score:.1f}".replace(".", "")
        dataset_id = (
            f"{cfg.method}_{self.template.config.template_id}"
            f"_score{min_score_str}_{ts}"
        )
        save_dir = self.output_dir / dataset_id
        save_dir.mkdir(parents=True, exist_ok=True)

        # --- train / val 分割 ---
        rng = random.Random(self.seed)
        shuffled = samples.copy()
        rng.shuffle(shuffled)
        split = int(len(shuffled) * _TRAIN_RATIO)
        train, val = shuffled[:split], shuffled[split:]

        self._write_jsonl(save_dir / "train.jsonl", train)
        self._write_jsonl(save_dir / "val.jsonl", val)

        # --- metadata ---
        scores = [s["score"] for s in samples]
        metadata = {
            "dataset_id": dataset_id,
            "created_at": ts,
            "template": self.template.to_dict(),
            "label_source": {
                "method":          cfg.method,
                "min_score":       cfg.min_score,
                "mcts_iterations": cfg.mcts_iterations,
            },
            "stats": {
                "n_steps_attempted": n_attempted,
                "n_samples":         len(samples),
                "n_train":           len(train),
                "n_val":             len(val),
                "n_skipped":         n_attempted - len(samples),
                "score_mean":        round(sum(scores) / len(scores), 3) if scores else 0.0,
                "score_min":         round(min(scores), 3) if scores else 0.0,
                "score_max":         round(max(scores), 3) if scores else 0.0,
            },
        }
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        stats = metadata["stats"]
        print(f"\n[DatasetBuilder] 保存先: {save_dir}")
        print(f"  train={stats['n_train']} | val={stats['n_val']} | skip={stats['n_skipped']}")
        if scores:
            print(
                f"  スコア: 平均={stats['score_mean']:.2f} "
                f"最小={stats['score_min']:.1f} "
                f"最大={stats['score_max']:.1f}"
            )
        return dataset_id

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
