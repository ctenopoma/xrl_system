"""PromptTemplate - 学習・推論で共有するプロンプト生成レイヤー。

FeatureSet と テンプレート文字列をバージョン管理し、
DatasetBuilder (学習) と InferenceEngine (推論) の両方から参照する。
テンプレートを変えれば学習・推論で同じフォーマットが保証される。

設定ファイル:
  configs/variable_descriptions.yaml  — センサー・操舵変数の説明テキスト
  configs/prompt_strategies.yaml      — few-shot 例示と CoT 思考ステップ
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Optional

import yaml

from modules.data_loader import STATE_COLS, ACTION_COLS, StepContext


# ------------------------------------------------------------------
# デフォルト YAML パス
# ------------------------------------------------------------------

_DEFAULT_DESC_YAML = (
    pathlib.Path(__file__).parent.parent / "configs" / "variable_descriptions.yaml"
)
_DEFAULT_STRATEGY_YAML = (
    pathlib.Path(__file__).parent.parent / "configs" / "prompt_strategies.yaml"
)


# ------------------------------------------------------------------
# YAML ローダー
# ------------------------------------------------------------------

def _load_var_descriptions(yaml_path: pathlib.Path | str | None = None) -> dict[str, str]:
    """variable_descriptions.yaml を読み込み {VARNAME: 説明} の辞書を返す。

    YAML 構造::

        sensor:
          altitude: "高度 (m)"
          ...
        action:
          aileron: "エルロン ..."
          ...

    返り値のキーは大文字化した変数名 (例: "ALTITUDE", "AILERON")。
    """
    path = pathlib.Path(yaml_path) if yaml_path else _DEFAULT_DESC_YAML
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    descriptions: dict[str, str] = {}
    for section in ("sensor", "action"):
        for var_name, desc in (data.get(section) or {}).items():
            descriptions[var_name.upper()] = str(desc)
    return descriptions


def _load_strategy_config(yaml_path: pathlib.Path | str | None = None) -> dict:
    """prompt_strategies.yaml を読み込んで返す。"""
    path = pathlib.Path(yaml_path) if yaml_path else _DEFAULT_STRATEGY_YAML
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ------------------------------------------------------------------
# センサー説明テキストのビルド (YAML 由来の説明を使用)
# ------------------------------------------------------------------

def _build_sensor_description(
    state_features: list[str],
    action_features: list[str],
    descriptions: dict[str, str],
) -> str:
    """使用するフィーチャーに対応する説明だけを組み立てる。

    テンプレート内では ``altitude:{ALTITUDE}`` のように書いておけば
    ``{ALTITUDE}`` が YAML 由来の説明テキストに置き換えられる。
    """
    sensor_lines = []
    for var in state_features:
        desc = descriptions.get(var.upper(), var)
        sensor_lines.append(f"- {var}: {desc}")

    action_lines = []
    for var in action_features:
        desc = descriptions.get(var.upper(), var)
        action_lines.append(f"- {var}: {desc}")

    parts = []
    if sensor_lines:
        parts.append("【センサーデータの意味】\n" + "\n".join(sensor_lines))
    if action_lines:
        parts.append("【操舵入力の意味】\n" + "\n".join(action_lines))
    return "\n\n".join(parts) + "\n"


# ------------------------------------------------------------------
# Few-shot ブロックのビルド (YAML 由来の例示を使用)
# ------------------------------------------------------------------

def _build_few_shot_block(examples: list[dict]) -> str:
    """few-shot 例示リストを1つのテキストブロックに組み立てる。

    Args:
        examples: YAML の few_shot.examples リスト。各要素は input/output キーを持つ。

    Returns:
        プロンプトに埋め込む文字列。例示がなければ空文字列。
    """
    if not examples:
        return ""

    lines = ["【参考例】"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"\n--- 例{i} ---")
        lines.append(ex.get("input", "").rstrip())
        lines.append("【説明】")
        lines.append(ex.get("output", "").rstrip())
    lines.append("\n--- 以上が参考例です ---")
    return "\n".join(lines)


# ------------------------------------------------------------------
# CoT サフィックスのビルド (YAML 由来の思考ステップを使用)
# ------------------------------------------------------------------

def _build_cot_suffix(cot_cfg: dict) -> str:
    """CoT 設定から思考ステップのサフィックスを組み立てる。

    Args:
        cot_cfg: YAML の cot セクション辞書。

    Returns:
        ユーザープロンプト末尾に付加するテキスト。
    """
    preamble = cot_cfg.get("preamble", "まず手順を追って論理的に推論してください。")
    steps: list[str] = cot_cfg.get("steps", [])
    conclusion_label = cot_cfg.get("conclusion_label", "【結論】")

    step_lines = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    if step_lines:
        return (
            f"\n\n{preamble}\n"
            f"{step_lines}\n\n"
            f"{conclusion_label}"
        )
    return f"\n\n{preamble}\n\n{conclusion_label}"


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
        few_shot_enabled:         few-shot 例示をプロンプトに含めるか
    """

    template_id: str = "v1_basic"
    state_features: list[str] = field(default_factory=lambda: list(STATE_COLS))
    action_features: list[str] = field(default_factory=lambda: list(ACTION_COLS))
    include_sensor_desc: bool = True
    output_length_hint: str = "200〜400字"
    prior_info_slot_enabled: bool = False
    few_shot_enabled: bool = False


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
        few_shot_enabled=False,
    ),
    # 高度・速度を除外してコンパクトに (特徴量削減実験用)
    "v1_combat_only": PromptTemplateConfig(
        template_id="v1_combat_only",
        state_features=["distance", "ata", "aspect_angle"],
        action_features=list(ACTION_COLS),
        include_sensor_desc=True,
        output_length_hint="200〜400字",
        prior_info_slot_enabled=False,
        few_shot_enabled=False,
    ),
    # 事前情報スロットあり (SySLLM 要約などを注入可能)
    "v2_with_prior": PromptTemplateConfig(
        template_id="v2_with_prior",
        state_features=list(STATE_COLS),
        action_features=list(ACTION_COLS),
        include_sensor_desc=True,
        output_length_hint="200〜400字",
        prior_info_slot_enabled=True,
        few_shot_enabled=False,
    ),
    # few-shot + センサー説明あり
    "v3_few_shot": PromptTemplateConfig(
        template_id="v3_few_shot",
        state_features=list(STATE_COLS),
        action_features=list(ACTION_COLS),
        include_sensor_desc=True,
        output_length_hint="200〜400字",
        prior_info_slot_enabled=False,
        few_shot_enabled=True,
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

        # CoT サフィックスを付加する場合 (InferenceEngine 経由)
        system, user = tpl.format_step(context, use_cot=True)

        # 事前情報を注入する場合
        tpl2 = PromptTemplate.from_preset("v2_with_prior")
        system, user = tpl2.format_step(context, prior_info="エピソード要約: ...")

        # few-shot を使う場合
        tpl3 = PromptTemplate.from_preset("v3_few_shot")
        system, user = tpl3.format_step(context)
    """

    def __init__(
        self,
        config: Optional[PromptTemplateConfig] = None,
        desc_yaml: pathlib.Path | str | None = None,
        strategy_yaml: pathlib.Path | str | None = None,
    ) -> None:
        self.config = config or PromptTemplateConfig()
        self._descriptions = _load_var_descriptions(desc_yaml)
        self._strategy = _load_strategy_config(strategy_yaml)

    @classmethod
    def from_preset(
        cls,
        name: str,
        desc_yaml: pathlib.Path | str | None = None,
        strategy_yaml: pathlib.Path | str | None = None,
    ) -> "PromptTemplate":
        """プリセット名からインスタンスを生成する。

        Args:
            name:          PRESETS のキー (e.g., "v1_basic", "v3_few_shot")
            desc_yaml:     変数説明 YAML のパス。省略時はデフォルトを使用。
            strategy_yaml: プロンプト戦略 YAML のパス。省略時はデフォルトを使用。

        Raises:
            KeyError: 未定義のプリセット名の場合
        """
        if name not in PRESETS:
            available = list(PRESETS.keys())
            raise KeyError(f"Unknown preset '{name}'. Available: {available}")
        return cls(config=PRESETS[name], desc_yaml=desc_yaml, strategy_yaml=strategy_yaml)

    # ------------------------------------------------------------------
    # 公開API
    # ------------------------------------------------------------------

    def format_step(
        self,
        context: StepContext,
        prior_info: str = "",
        use_cot: bool = False,
    ) -> tuple[str, str]:
        """StepContext を (system_prompt, user_prompt) に変換する。

        Args:
            context:    DataLoader.get_step_context() の返り値
            prior_info: 推論時に注入する事前情報 (エピソード要約など)
                        config.prior_info_slot_enabled=False の場合は無視される
            use_cot:    True の場合、CoT 思考ステップをユーザープロンプト末尾に付加する

        Returns:
            (system_prompt, user_prompt) のタプル
        """
        system = self._build_system()
        user = self._build_user(context, prior_info)
        if use_cot:
            user += _build_cot_suffix(self._strategy.get("cot", {}))
        return system, user

    def build_cot_suffix(self) -> str:
        """YAML 設定から CoT サフィックスを生成して返す。

        InferenceEngine が strategy=COT のとき呼び出す。
        """
        return _build_cot_suffix(self._strategy.get("cot", {}))

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
            "few_shot_enabled": cfg.few_shot_enabled,
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
            base += _build_sensor_description(
                self.config.state_features,
                self.config.action_features,
                self._descriptions,
            )

        # few_shot placement=after_desc の場合、システムプロンプト末尾に挿入
        if self.config.few_shot_enabled:
            fs_cfg = self._strategy.get("few_shot", {})
            if fs_cfg.get("placement", "before_state") == "after_desc":
                block = _build_few_shot_block(fs_cfg.get("examples", []))
                if block:
                    base += "\n" + block + "\n"

        return base

    def _build_user(self, context: StepContext, prior_info: str) -> str:
        parts: list[str] = []

        # 事前情報スロット
        if self.config.prior_info_slot_enabled and prior_info:
            parts.append(f"【事前情報】\n{prior_info}")

        # few_shot placement=before_state の場合、センサー状態ブロックの前に挿入
        if self.config.few_shot_enabled:
            fs_cfg = self._strategy.get("few_shot", {})
            if fs_cfg.get("placement", "before_state") == "before_state":
                block = _build_few_shot_block(fs_cfg.get("examples", []))
                if block:
                    parts.append(block)

        parts.append(f"【分析対象ステップ: Step {context['step']}】")
        parts.append(f"【センサー状態】\n{self.format_state(context['state'])}")
        parts.append(f"【操舵入力】\n{self.format_action(context['action'])}")
        parts.append(
            "この操舵入力をエージェントが選択した理由を、センサーデータの数値に基づいて説明してください。"
            f"{self.config.output_length_hint}程度でまとめてください。"
        )

        return "\n\n".join(parts)
