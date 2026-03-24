"""InferenceEngine - PromptTemplate + LLMClient を統合した推論レイヤー。

学習なし (外部 API) と 学習あり (ローカル LoRA) の両方に対応できる設計。
現時点では外部 API モードのみ実装。
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from modules.data_loader import StepContext
from modules.llm_client import LLMClient
from modules.prompt_template import PromptTemplate


# ------------------------------------------------------------------
# プロンプト戦略
# ------------------------------------------------------------------

class PromptingStrategy(str, Enum):
    """推論時のプロンプト戦略。

    ZERO_SHOT:  テンプレートそのまま
    COT:        "Chain-of-Thought で推論してから結論をまとめてください" を追加
    """
    ZERO_SHOT = "zero_shot"
    COT = "cot"


_COT_SUFFIX = (
    "\n\nまず手順を追って論理的に推論し (Chain-of-Thought)、"
    "最後に結論を簡潔にまとめてください。"
)


# ------------------------------------------------------------------
# InferenceEngine
# ------------------------------------------------------------------

class InferenceEngine:
    """PromptTemplate + LLMClient を組み合わせた推論エンジン。

    使い方::

        tpl    = PromptTemplate.from_preset("v1_basic")
        engine = InferenceEngine(llm, tpl, strategy=PromptingStrategy.ZERO_SHOT)
        explanation = engine.generate(context)

        # 事前情報を注入する場合 (v2_with_prior テンプレートが必要)
        engine2 = InferenceEngine(llm, PromptTemplate.from_preset("v2_with_prior"))
        explanation = engine2.generate(context, prior_info="エピソード要約: ...")
    """

    def __init__(
        self,
        llm: LLMClient,
        template: PromptTemplate,
        strategy: PromptingStrategy = PromptingStrategy.ZERO_SHOT,
    ) -> None:
        self.llm = llm
        self.template = template
        self.strategy = strategy

    # ------------------------------------------------------------------
    # 公開API
    # ------------------------------------------------------------------

    def generate(
        self,
        context: StepContext,
        prior_info: str = "",
    ) -> str:
        """StepContext から説明文を生成する。

        Args:
            context:    DataLoader.get_step_context() の返り値
            prior_info: 事前情報 (template.config.prior_info_slot_enabled=True の時のみ有効)

        Returns:
            生成された説明テキスト
        """
        system, user = self.template.format_step(context, prior_info)

        if self.strategy == PromptingStrategy.COT:
            user += _COT_SUFFIX

        return self.llm.simple_prompt(system, user)

    def to_dict(self) -> dict:
        """設定を dict で返す (結果 JSON への記録用)。"""
        return {
            "template": self.template.to_dict(),
            "strategy": self.strategy.value,
            "model": self.llm.model,
        }
