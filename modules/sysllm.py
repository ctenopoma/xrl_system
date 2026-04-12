"""SySLLM アプローチ。

論文 "SySLLM: Synthesized Summary using LLMs" (arXiv:2503.10509) に基づく実装。

処理フロー:
  Phase 1 — 経験収集・キャプション
    各ステップの (observation, action) を Cobs / Cact でテキスト化し
    Textual Experience Buffer (TEB) に蓄積する。

  Phase 2 — 階層的要約 + コンセンサス選択
    TEB がトークン予算 κ 以内なら K 候補を生成して埋め込みの重心最近傍を選ぶ。
    超過する場合は TEB を M 分割して再帰的に要約し最後に集約する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from modules.data_loader import DataLoader
from modules.llm_client import LLMClient

# -----------------------------------------------------------------------
# 定数
# -----------------------------------------------------------------------

# Phase 2 で生成するサマリー候補数
DEFAULT_K_CANDIDATES: int = 5

# TEB のトークン予算 (文字数で近似; 1 トークン ≈ 3〜4 文字)
DEFAULT_KAPPA: int = 12_000

# 出力4項目のキー
SUMMARY_KEYS = [
    "tactical_approach",
    "situational_adaptation",
    "inefficiencies",
    "overall_summary",
]

_SUMMARIZE_SYSTEM = """\
あなたは空中戦シミュレーターの戦術分析の専門家です。
強化学習エージェントの行動ログ（自然言語に変換済み）を受け取り、
以下の4つの観点から詳細に分析してください。

【出力フォーマット】
必ず以下のヘッダーで4つのセクションに分けて出力してください。

## 1. 戦術的アプローチ (tactical_approach)
## 2. 状況への適応 (situational_adaptation)
## 3. 非効率性と弱点 (inefficiencies)
## 4. 総合要約 (overall_summary)
"""

_AGGREGATE_SYSTEM = """\
あなたは空中戦シミュレーターの戦術分析の専門家です。
複数のサブサマリーを受け取り、それらを統合して
以下の4つのセクションにまとめてください。

【出力フォーマット】
## 1. 戦術的アプローチ (tactical_approach)
## 2. 状況への適応 (situational_adaptation)
## 3. 非効率性と弱点 (inefficiencies)
## 4. 総合要約 (overall_summary)
"""


# -----------------------------------------------------------------------
# データクラス: TEB エントリ
# -----------------------------------------------------------------------

@dataclass
class TEBEntry:
    """Textual Experience Buffer の1ステップ分のエントリ。"""
    step: int
    obs_caption: str   # Cobs(ot)
    act_caption: str   # Cact(at)
    reward: float = 0.0
    episode_id: int = 0


# -----------------------------------------------------------------------
# キャプション関数 Cobs / Cact
# -----------------------------------------------------------------------

def caption_obs(row: pd.Series) -> str:
    """観測をテキストに変換する (Cobs)。

    論文における captioning function の観測側。
    """
    alt = row["altitude"]
    spd = row["speed"]
    dist = row["distance"]
    ata = row["ata"]
    aa = row["aspect_angle"]

    # 高度帯
    if alt < 1000:
        alt_desc = "超低高度"
    elif alt < 3000:
        alt_desc = "低高度"
    elif alt < 6000:
        alt_desc = "中高度"
    else:
        alt_desc = "高高度"

    # 速度帯
    if spd < 200:
        spd_desc = "低速"
    elif spd < 300:
        spd_desc = "巡航速度"
    else:
        spd_desc = "高速"

    # 距離帯
    if dist < 1000:
        dist_desc = "至近距離"
    elif dist < 3000:
        dist_desc = "近距離"
    elif dist < 8000:
        dist_desc = "中距離"
    else:
        dist_desc = "遠距離"

    # 攻撃機会
    attack_opp = "（攻撃機会あり）" if ata < 45 else ""

    return (
        f"{alt_desc}({alt:.0f}m)・{spd_desc}({spd:.0f}kt)・"
        f"{dist_desc}({dist:.0f}m)で飛行中。"
        f"ATA={ata:.1f}°{attack_opp}、AspectAngle={aa:.1f}°。"
    )


def caption_act(row: pd.Series) -> str:
    """行動をテキストに変換する (Cact)。

    論文における captioning function の行動側。
    """
    ail = row["aileron"]
    ele = row["elevator"]
    thr = row["throttle"]

    # エルロン
    if ail > 0.3:
        ail_desc = f"右ロール(強:{ail:.2f})"
    elif ail > 0.1:
        ail_desc = f"右ロール(弱:{ail:.2f})"
    elif ail < -0.3:
        ail_desc = f"左ロール(強:{ail:.2f})"
    elif ail < -0.1:
        ail_desc = f"左ロール(弱:{ail:.2f})"
    else:
        ail_desc = "ロール中立"

    # エレベータ
    if ele > 0.3:
        ele_desc = f"引き起こし(強:{ele:.2f})"
    elif ele > 0.1:
        ele_desc = f"引き起こし(弱:{ele:.2f})"
    elif ele < -0.3:
        ele_desc = f"押し下げ(強:{ele:.2f})"
    elif ele < -0.1:
        ele_desc = f"押し下げ(弱:{ele:.2f})"
    else:
        ele_desc = "ピッチ中立"

    # スロットル
    if thr > 0.8:
        thr_desc = f"全開({thr:.2f})"
    elif thr > 0.5:
        thr_desc = f"中出力({thr:.2f})"
    else:
        thr_desc = f"低出力({thr:.2f})"

    return f"{ail_desc}、{ele_desc}、スロットル{thr_desc}。"


# -----------------------------------------------------------------------
# SySLLM 本体
# -----------------------------------------------------------------------

class SySLLM:
    """論文準拠の SySLLM 実装。

    Phase 1: 全ステップを Cobs/Cact でキャプションして TEB を構築する。
    Phase 2: TEB を階層的に要約し、コンセンサス選択で最終サマリーを決定する。

    使い方::

        sysllm = SySLLM(llm, loader)
        result = sysllm.analyze()
        print(result["overall_summary"])
    """

    def __init__(
        self,
        llm: LLMClient,
        loader: DataLoader,
        k_candidates: int = DEFAULT_K_CANDIDATES,
        kappa: int = DEFAULT_KAPPA,
    ) -> None:
        self.llm = llm
        self.loader = loader
        self.k_candidates = k_candidates
        self.kappa = kappa

    # ------------------------------------------------------------------
    # 公開API
    # ------------------------------------------------------------------

    def analyze(self, episode_id: int = 0) -> dict:
        """エピソードを分析してサマリーを返す。

        Args:
            episode_id: エピソード識別子 (複数エピソードを区別するための番号)

        Returns:
            {
              "tactical_approach":       str,
              "situational_adaptation":  str,
              "inefficiencies":          str,
              "overall_summary":         str,
              "raw_response":            str,   # 最終的な生テキスト
              "n_teb_entries":           int,   # TEB エントリ数
              "n_candidates":            int,   # 生成した候補数
            }
        """
        # Phase 1: TEB 構築
        teb = self._build_teb(episode_id)

        # Phase 2: 階層的要約 + コンセンサス選択
        raw = self._hierarchical_summarize(teb)

        parsed = self._parse_response(raw)
        parsed["raw_response"] = raw
        parsed["n_teb_entries"] = len(teb)
        parsed["n_candidates"] = self.k_candidates
        return parsed

    # ------------------------------------------------------------------
    # Phase 1: TEB 構築
    # ------------------------------------------------------------------

    def _build_teb(self, episode_id: int = 0) -> list[TEBEntry]:
        """DataFrame の全ステップを Cobs/Cact でキャプションして TEB を返す。

        論文 Algorithm 1 に対応: 全ステップを処理し、キーフレームフィルタは行わない。
        """
        df = self.loader.load()
        teb: list[TEBEntry] = []
        for _, row in df.iterrows():
            entry = TEBEntry(
                step=int(row["step"]),
                obs_caption=caption_obs(row),
                act_caption=caption_act(row),
                episode_id=episode_id,
            )
            teb.append(entry)
        return teb

    # ------------------------------------------------------------------
    # Phase 2: 階層的要約
    # ------------------------------------------------------------------

    def _hierarchical_summarize(self, teb: list[TEBEntry]) -> str:
        """TEB を受け取り、最終サマリーテキストを返す。

        論文 Algorithm 2 に対応:
          - |TEB| <= κ: K 候補生成 → コンセンサス選択
          - |TEB| >  κ: M 分割 → 各サブセットを再帰要約 → 集約
        """
        teb_text = self._teb_to_text(teb)

        if len(teb_text) <= self.kappa:
            return self._summarize_with_consensus(teb_text)

        # 分割数: 各サブセットが kappa 以下になるよう設定
        m = math.ceil(len(teb_text) / self.kappa)
        chunk_size = math.ceil(len(teb) / m)
        sub_summaries: list[str] = []
        for i in range(0, len(teb), chunk_size):
            sub_teb = teb[i : i + chunk_size]
            sub_summaries.append(self._hierarchical_summarize(sub_teb))

        return self._aggregate_summaries(sub_summaries)

    def _summarize_with_consensus(self, teb_text: str) -> str:
        """K 候補サマリーを生成し、埋め込み重心に最近傍のものを返す。

        論文 Section 4.3 のコンセンサス選択に対応。
        """
        user_prompt = (
            "以下は強化学習エージェントのフライト軌跡ログです（全ステップのテキスト表現）。\n\n"
            f"{teb_text}\n\n"
            "上記のデータを詳しく分析し、指定されたフォーマットで4つのセクションを出力してください。"
        )

        # K 候補を生成
        candidates: list[str] = []
        for _ in range(self.k_candidates):
            resp = self.llm.simple_prompt(
                _SUMMARIZE_SYSTEM,
                user_prompt,
                temperature=0.5,
            )
            candidates.append(resp)

        if len(candidates) == 1:
            return candidates[0]

        # 埋め込みによるコンセンサス選択
        try:
            embeddings = self.llm.embed(candidates)          # (K, dim)
            centroid = embeddings.mean(axis=0)               # (dim,)
            diffs = embeddings - centroid                    # (K, dim)
            dists = np.linalg.norm(diffs, axis=1)           # (K,)
            best_idx = int(np.argmin(dists))
            return candidates[best_idx]
        except Exception:
            # 埋め込みが使えない場合は先頭候補を返す
            return candidates[0]

    def _aggregate_summaries(self, sub_summaries: list[str]) -> str:
        """複数のサブサマリーを集約して最終サマリーを返す。"""
        combined = "\n\n---\n\n".join(
            f"[サブサマリー {i+1}]\n{s}" for i, s in enumerate(sub_summaries)
        )
        user_prompt = (
            "以下の複数のサブサマリーを統合し、"
            "指定されたフォーマットで4つのセクションにまとめてください。\n\n"
            f"{combined}"
        )
        return self.llm.simple_prompt(_AGGREGATE_SYSTEM, user_prompt, temperature=0.3)

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    @staticmethod
    def _teb_to_text(teb: list[TEBEntry]) -> str:
        """TEB エントリリストをプロンプト用テキストに変換する。"""
        lines: list[str] = []
        for e in teb:
            lines.append(
                f"Step {e.step} [ep={e.episode_id}]: {e.obs_caption} → {e.act_caption}"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """LLM の応答テキストを4つのキーに分割してパースする。"""
        result = {key: "" for key in SUMMARY_KEYS}

        header_map = [
            ("## 1.", "tactical_approach"),
            ("## 2.", "situational_adaptation"),
            ("## 3.", "inefficiencies"),
            ("## 4.", "overall_summary"),
        ]

        lines = raw.split("\n")
        current_key: Optional[str] = None
        buffer: list[str] = []

        for line in lines:
            matched = False
            for header_prefix, key in header_map:
                if line.strip().startswith(header_prefix):
                    if current_key:
                        result[current_key] = "\n".join(buffer).strip()
                    current_key = key
                    buffer = []
                    matched = True
                    break
            if not matched and current_key:
                buffer.append(line)

        if current_key:
            result[current_key] = "\n".join(buffer).strip()

        return result
