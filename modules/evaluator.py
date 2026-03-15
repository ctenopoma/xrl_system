"""LLM-as-a-Judge 自動評価モジュール。

各手法で生成された説明文を、別の LLM インスタンスが Soundness/Fidelity の
2軸で自動採点する。
"""

from __future__ import annotations

import json
import re
from typing import Optional

from modules.data_loader import StepContext
from modules.llm_client import LLMClient

# 評価結果の型エイリアス
EvalResult = dict  # {"soundness": int, "fidelity": int, "reason": str}

_JUDGE_SYSTEM = """\
あなたは強化学習エージェントの行動説明品質を評価する審判 (Judge) です。
提示された説明を、以下の2つの指標で採点してください。

【評価指標（各0〜2の整数）】

■ Soundness（論理的妥当性）
  環境の物理的メカニズムと論理的に矛盾していないか。エージェントの目的に合致しているか。
  0 = 明らかな物理的・論理的矛盾が存在する
  1 = 一部不正確または論理的飛躍がある
  2 = 完全に物理的・論理的に妥当

■ Fidelity（忠実性）
  実際の状態数値（距離や角度の変化）に基づいた正しい因果関係が説明されているか。
  無関係な要素を捏造していないか。
  0 = 実際のデータを無視または捏造している
  1 = 部分的にデータに基づいているが不正確な点がある
  2 = 実際のデータに完全に忠実

【出力フォーマット】
必ず以下の JSON 形式のみで出力してください（説明等は reason フィールドに含める）:
{"soundness": int, "fidelity": int, "reason": str}
"""

_MAX_EVAL_RETRIES = 3


class EpisodeEvaluator:
    """LLM-as-a-Judge による自動評価モジュール。

    使い方::

        evaluator = EpisodeEvaluator(llm)
        result = evaluator.evaluate(explanation, context=ctx)
        print(result["soundness"], result["fidelity"])

        # 複数の説明をまとめて評価
        results = evaluator.evaluate_batch(method_results, context=ctx)
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    # ------------------------------------------------------------------
    # 公開API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        explanation: str,
        context: Optional[StepContext] = None,
        trajectory_text: str = "",
    ) -> EvalResult:
        """説明文を採点して EvalResult を返す。

        Args:
            explanation:     採点対象の説明テキスト
            context:         参照するステップコンテキスト (局所評価時)
            trajectory_text: 参照する軌跡テキスト (全体評価時)

        Returns:
            {"soundness": int(0-2), "fidelity": int(0-2), "reason": str}

        Raises:
            ValueError: JSON パースに MAX_EVAL_RETRIES 回失敗した場合
        """
        system, user = self._build_judge_prompt(explanation, context, trajectory_text)

        last_error: Optional[Exception] = None
        for attempt in range(_MAX_EVAL_RETRIES):
            try:
                raw = self.llm.simple_prompt(
                    system, user, response_format={"type": "json_object"}
                )
                result = self._parse_json_response(raw)
                # 値域チェック
                result["soundness"] = int(max(0, min(2, result.get("soundness", 0))))
                result["fidelity"] = int(max(0, min(2, result.get("fidelity", 0))))
                if "reason" not in result:
                    result["reason"] = ""
                return result
            except (ValueError, KeyError, TypeError) as e:
                last_error = e
                print(f"[Evaluator] JSON パース失敗 (試行 {attempt + 1}/{_MAX_EVAL_RETRIES}): {e}")

        raise ValueError(
            f"評価 JSON のパースに {_MAX_EVAL_RETRIES} 回失敗しました: {last_error}"
        )

    def evaluate_batch(
        self,
        results: list[dict],
        context: Optional[StepContext] = None,
        trajectory_text: str = "",
    ) -> list[dict]:
        """複数の説明をまとめて採点する。

        Args:
            results: 各モジュールの出力辞書リスト。各辞書に "explanation" キーが必要。

        Returns:
            入力 results に "eval" キー (EvalResult) を追加したリスト
        """
        evaluated = []
        for item in results:
            explanation = item.get("explanation", "")
            if not explanation:
                item["eval"] = {"soundness": 0, "fidelity": 0, "reason": "説明が空です"}
            else:
                item["eval"] = self.evaluate(explanation, context, trajectory_text)
            evaluated.append(item)
        return evaluated

    # ------------------------------------------------------------------
    # 非公開ヘルパー
    # ------------------------------------------------------------------

    def _build_judge_prompt(
        self,
        explanation: str,
        context: Optional[StepContext],
        trajectory_text: str,
    ) -> tuple[str, str]:
        """(system_prompt, user_prompt) を返す。"""
        reference_section = ""

        if context:
            state_lines = "\n".join(
                f"  {k}: {v:.2f}" for k, v in context["state"].items()
            )
            action_lines = "\n".join(
                f"  {k}: {v:.2f}" for k, v in context["action"].items()
            )
            reference_section += (
                f"\n【参照データ: Step {context['step']}】\n"
                f"センサー状態:\n{state_lines}\n"
                f"操舵入力:\n{action_lines}\n"
            )

        if trajectory_text:
            # 全体評価時は軌跡の一部を参照として渡す
            preview = trajectory_text[:1000] + ("..." if len(trajectory_text) > 1000 else "")
            reference_section += f"\n【参照データ: 軌跡ログ (抜粋)】\n{preview}\n"

        user = (
            f"{reference_section}\n"
            f"【評価対象の説明】\n{explanation}\n\n"
            "上記の説明を採点し、JSON 形式で出力してください。"
        )
        return _JUDGE_SYSTEM, user

    def _parse_json_response(self, text: str) -> EvalResult:
        """LLM 応答から JSON を抽出してパースする。

        戦略:
            1. テキスト全体を json.loads() でパース
            2. 失敗時は ```json...``` ブロックを正規表現で抽出
            3. 失敗時は {...} を正規表現で抽出

        Raises:
            ValueError: すべての抽出方法が失敗した場合
        """
        # 試行1: そのままパース
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 試行2: ```json ... ``` ブロックを抽出
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 試行3: 最初の {...} を抽出
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"JSON パース失敗: {text[:300]}")
