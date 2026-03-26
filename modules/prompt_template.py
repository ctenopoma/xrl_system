"""PromptTemplate - 学習・推論で共有するプロンプト生成レイヤー。

FeatureSet と テンプレート文字列をバージョン管理し、
DatasetBuilder (学習) と InferenceEngine (推論) の両方から参照する。
テンプレートを変えれば学習・推論で同じフォーマットが保証される。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from modules.data_loader import STATE_COLS, ACTION_COLS, StepContext


# ------------------------------------------------------------------
# センサー説明テキスト (テンプレートに埋め込むメタ知識)
# ------------------------------------------------------------------

_SENSOR_DESCRIPTION = """\
【センサーデータの意味】
- altitude: 高度 (m)
- speed: 速度 (kt)
- distance: 敵機との距離 (m)
- ata: 自機から敵機へのATA角度 (°) ※45°未満が攻撃チャンス
- aspect_angle: 敵機から自機への角度 (°)

【操舵入力の意味】
- aileron: エルロン (-1=左ロール, +1=右ロール)
- elevator: エレベータ (-1=機首下げ, +1=機首上げ)
- throttle: スロットル (0=アイドル, 1=最大推力)
"""


# ------------------------------------------------------------------
# 設定データクラス
# ------------------------------------------------------------------

@dataclass
class PromptTemplateConfig:
    """PromptTemplate の設定。YAML や argparse から渡す想定。

    Attributes:
        template_id:              バージョン識別子 (結果 JSON に記録される)
        state_features:           使用するセンサー特徴量のリスト
        action_features:          使用する操舵入力のリスト
        include_sensor_desc:      センサーの意味説明をプロンプトに含めるか
        output_length_hint:       出力文字数の指示
        prior_info_slot_enabled:  事前情報スロットを有効にするか
    """

    template_id: str = "v1_basic"
    state_features: list[str] = field(default_factory=lambda: list(STATE_COLS))
    action_features: list[str] = field(default_factory=lambda: list(ACTION_COLS))
    include_sensor_desc: bool = True
    output_length_hint: str = "200〜400字"
    prior_info_slot_enabled: bool = False


# ------------------------------------------------------------------
# プリセット (コード上の "named versions")
# ------------------------------------------------------------------

PRESETS: dict[str, PromptTemplateConfig] = {
    # 全特徴量・センサー説明あり
    "v1_basic": PromptTemplateConfig(
        template_id="v1_basic",
        state_features=list(STATE_COLS),
        action_features=list(ACTION_COLS),
        include_sensor_desc=True,
        output_length_hint="200〜400字",
        prior_info_slot_enabled=False,
    ),
    # 高度・速度を除外してコンパクトに (特徴量削減実験用)
    "v1_combat_only": PromptTemplateConfig(
        template_id="v1_combat_only",
        state_features=["distance", "ata", "aspect_angle"],
        action_features=list(ACTION_COLS),
        include_sensor_desc=True,
        output_length_hint="200〜400字",
        prior_info_slot_enabled=False,
    ),
    # 事前情報スロットあり (SySLLM 要約などを注入可能)
    "v2_with_prior": PromptTemplateConfig(
        template_id="v2_with_prior",
        state_features=list(STATE_COLS),
        action_features=list(ACTION_COLS),
        include_sensor_desc=True,
        output_length_hint="200〜400字",
        prior_info_slot_enabled=True,
    ),
}


# ------------------------------------------------------------------
# PromptTemplate 本体
# ------------------------------------------------------------------

class PromptTemplate:
    """feature set とテンプレート文字列を管理する共通レイヤー。

    使い方::

        tpl = PromptTemplate.from_preset("v1_basic")
        system, user = tpl.format_step(context)

        # 事前情報を注入する場合
        tpl2 = PromptTemplate.from_preset("v2_with_prior")
        system, user = tpl2.format_step(context, prior_info="エピソード要約: ...")
    """

    def __init__(self, config: Optional[PromptTemplateConfig] = None) -> None:
        self.config = config or PromptTemplateConfig()

    @classmethod
    def from_preset(cls, name: str) -> "PromptTemplate":
        """プリセット名からインスタンスを生成する。

        Args:
            name: PRESETS のキー (e.g., "v1_basic", "v2_with_prior")

        Raises:
            KeyError: 未定義のプリセット名の場合
        """
        if name not in PRESETS:
            available = list(PRESETS.keys())
            raise KeyError(f"Unknown preset '{name}'. Available: {available}")
        return cls(config=PRESETS[name])

    # ------------------------------------------------------------------
    # 公開API
    # ------------------------------------------------------------------

    def format_step(
        self,
        context: StepContext,
        prior_info: str = "",
    ) -> tuple[str, str]:
        """StepContext を (system_prompt, user_prompt) に変換する。

        Args:
            context:    DataLoader.get_step_context() の返り値
            prior_info: 推論時に注入する事前情報 (エピソード要約など)
                        config.prior_info_slot_enabled=False の場合は無視される

        Returns:
            (system_prompt, user_prompt) のタプル
        """
        system = self._build_system()
        user = self._build_user(context, prior_info)
        return system, user

    def format_state(self, state: dict) -> str:
        """設定された state_features のみをテキスト化する。"""
        return "\n".join(
            f"  {k}: {v:.2f}"
            for k, v in state.items()
            if k in self.config.state_features
        )

    def format_action(self, action: dict) -> str:
        """設定された action_features のみをテキスト化する。"""
        return "\n".join(
            f"  {k}: {v:.2f}"
            for k, v in action.items()
            if k in self.config.action_features
        )

    def to_dict(self) -> dict:
        """設定を dict で返す (結果 JSON への記録用)。"""
        cfg = self.config
        return {
            "template_id": cfg.template_id,
            "state_features": cfg.state_features,
            "action_features": cfg.action_features,
            "include_sensor_desc": cfg.include_sensor_desc,
            "output_length_hint": cfg.output_length_hint,
            "prior_info_slot_enabled": cfg.prior_info_slot_enabled,
        }

    # ------------------------------------------------------------------
    # 内部ビルダー
    # ------------------------------------------------------------------

    def _build_system(self) -> str:
        base = (
            "あなたは空中戦シミュレーターの行動分析専門家です。\n"
            "強化学習エージェントが特定のステップで行った操舵入力の理由を、"
            "センサーデータに基づいて論理的に説明してください。\n"
        )
        if self.config.include_sensor_desc:
            base += _SENSOR_DESCRIPTION
        return base

    def _build_user(self, context: StepContext, prior_info: str) -> str:
        parts: list[str] = []

        # 事前情報スロット
        if self.config.prior_info_slot_enabled and prior_info:
            parts.append(f"【事前情報】\n{prior_info}")

        parts.append(f"【分析対象ステップ: Step {context['step']}】")
        parts.append(f"【センサー状態】\n{self.format_state(context['state'])}")
        parts.append(f"【操舵入力】\n{self.format_action(context['action'])}")
        parts.append(
            "この操舵入力をエージェントが選択した理由を、センサーデータの数値に基づいて説明してください。"
            f"{self.config.output_length_hint}程度でまとめてください。"
        )

        return "\n\n".join(parts)
