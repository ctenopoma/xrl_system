"""SySLLM アプローチ。

間引きされた軌跡全体を入力とし、エージェントの全体的な戦術・弱点をトップダウンで一括要約する。
"""

from __future__ import annotations

from modules.data_loader import DataLoader
from modules.llm_client import LLMClient

# 出力4項目のキー
SUMMARY_KEYS = [
    "tactical_approach",
    "situational_adaptation",
    "inefficiencies",
    "overall_summary",
]

_SYSTEM_PROMPT = """\
あなたは空中戦シミュレーターの戦術分析の専門家です。
強化学習エージェントがフライトシミュレーターで実行した一連の行動ログを受け取り、
以下の4つの観点から詳細に分析してください。

【出力フォーマット】
必ず以下のヘッダーで4つのセクションに分けて出力してください。

## 1. 戦術的アプローチ (tactical_approach)
## 2. 状況への適応 (situational_adaptation)
## 3. 非効率性と弱点 (inefficiencies)
## 4. 総合要約 (overall_summary)

【データの説明】
- altitude: 高度 (m)
- speed: 速度 (kt)
- distance: 敵機との距離 (m)
- ata: 自機から敵機へのATA角度 (°) ※45°未満が攻撃チャンス
- aspect_angle: 敵機から自機への角度 (°)
- aileron: エルロン操舵 (-1〜1)
- elevator: エレベータ操舵 (-1〜1)
- throttle: スロットル (0〜1)
"""


class SySLLM:
    """エピソード全体の戦術をトップダウンで一括要約するモジュール。

    使い方::

        sysllm = SySLLM(llm, loader)
        result = sysllm.analyze()
        print(result["overall_summary"])
    """

    def __init__(self, llm: LLMClient, loader: DataLoader) -> None:
        self.llm = llm
        self.loader = loader

    # ------------------------------------------------------------------
    # 公開API
    # ------------------------------------------------------------------

    def analyze(self, max_rows: int = 50) -> dict:
        """キーフレームを抽出して LLM に要約させる。

        Args:
            max_rows: プロンプトに渡す最大行数

        Returns:
            {
              "tactical_approach":       str,
              "situational_adaptation":  str,
              "inefficiencies":          str,
              "overall_summary":         str,
              "raw_response":            str,   # デバッグ用生テキスト
              "n_keyframes":             int,   # 使用したキーフレーム数
            }
        """
        df = self.loader.load()
        keyframes = self.loader.filter_keyframes(df)
        trajectory_text = self.loader.to_trajectory_text(keyframes, max_rows=max_rows)

        system_prompt, user_prompt = self._build_prompt(trajectory_text)
        raw = self.llm.simple_prompt(system_prompt, user_prompt)

        parsed = self._parse_response(raw)
        parsed["raw_response"] = raw
        parsed["n_keyframes"] = len(keyframes)
        return parsed

    # ------------------------------------------------------------------
    # 非公開ヘルパー
    # ------------------------------------------------------------------

    def _build_prompt(self, trajectory_text: str) -> tuple[str, str]:
        """(system_prompt, user_prompt) のタプルを返す。"""
        user_prompt = (
            "以下は強化学習エージェントのフライト軌跡ログ（キーフレームのみ抽出済み）です。\n\n"
            f"{trajectory_text}\n\n"
            "上記のデータを詳しく分析し、指定されたフォーマットで4つのセクションを出力してください。"
        )
        return _SYSTEM_PROMPT, user_prompt

    def _parse_response(self, raw: str) -> dict:
        """LLM の応答テキストを4つのキーに分割してパースする。

        ヘッダー行を区切りとして各セクションを抽出する。
        パースに失敗した場合はキーに空文字列を設定する。
        """
        result = {key: "" for key in SUMMARY_KEYS}

        # ヘッダーパターンと対応するキー
        header_map = [
            ("## 1.", "tactical_approach"),
            ("## 2.", "situational_adaptation"),
            ("## 3.", "inefficiencies"),
            ("## 4.", "overall_summary"),
        ]

        lines = raw.split("\n")
        current_key: str | None = None
        buffer: list[str] = []

        for line in lines:
            matched = False
            for header_prefix, key in header_map:
                if line.strip().startswith(header_prefix):
                    # 前のセクションを保存
                    if current_key:
                        result[current_key] = "\n".join(buffer).strip()
                    current_key = key
                    buffer = []
                    matched = True
                    break
            if not matched and current_key:
                buffer.append(line)

        # 最後のセクションを保存
        if current_key:
            result[current_key] = "\n".join(buffer).strip()

        return result
