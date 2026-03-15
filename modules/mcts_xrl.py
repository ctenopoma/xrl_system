"""CoT / MCTS-XRL アプローチ。

モード1 (CoT): 単一プロンプトによる基本的な推論。
モード4 (MCTS): Generator/Critic/Refiner/Evaluator の4役が協調して
               反復的に説明を自己改善する。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from modules.data_loader import DataLoader, StepContext
from modules.llm_client import LLMClient

DEFAULT_ITERATIONS = 4
UCB_CONSTANT = math.sqrt(2)  # UCB1 の探索定数

_STATE_DESCRIPTION = """\
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


@dataclass
class MCTSNode:
    """MCTS の1ノード = 1つの説明テキスト。"""

    explanation: str
    parent: Optional["MCTSNode"] = field(default=None, repr=False)
    children: list["MCTSNode"] = field(default_factory=list, repr=False)
    q_value: float = 0.0       # Evaluator から付与されるスコア (0.0〜4.0)
    visits: int = 0
    critic_feedback: str = ""  # このノードへの批判テキスト

    def ucb1(self, parent_visits: int) -> float:
        """UCB1 スコアを計算する。未訪問ノードは +inf を返す。"""
        if self.visits == 0:
            return float("inf")
        return self.q_value / self.visits + UCB_CONSTANT * math.sqrt(
            math.log(parent_visits) / self.visits
        )


class MCTSXRL:
    """CoT および MCTS ベースの局所説明モジュール。

    使い方::

        mcts = MCTSXRL(llm, loader, iterations=4)
        result = mcts.explain_cot(step=150)    # CoT モード
        result = mcts.explain_mcts(step=150)   # MCTS モード
    """

    def __init__(
        self,
        llm: LLMClient,
        loader: DataLoader,
        iterations: int = DEFAULT_ITERATIONS,
    ) -> None:
        self.llm = llm
        self.loader = loader
        self.iterations = iterations

    # ------------------------------------------------------------------
    # 公開API
    # ------------------------------------------------------------------

    def explain_cot(self, step: int) -> dict:
        """CoT モードの説明生成 (MCTS 不使用、単一プロンプト)。

        Args:
            step: 分析対象ステップ番号

        Returns:
            {"step": int, "explanation": str, "context": StepContext}
        """
        context = self.loader.get_step_context(step)
        explanation = self._generator(context)
        return {"step": step, "explanation": explanation, "context": context}

    def explain_mcts(self, step: int) -> dict:
        """MCTS モードの説明生成。

        Generator → Critic → Refiner → Evaluator を iterations 回繰り返し、
        最高 Q 値の説明を返す。

        Args:
            step: 分析対象ステップ番号

        Returns:
            {
              "step":         int,
              "explanation":  str,       # 最高 Q 値ノードのテキスト
              "best_q":       float,
              "iterations":   int,
              "tree_summary": list[dict],   # 各ノードの概要
              "context":      StepContext,
            }
        """
        context = self.loader.get_step_context(step)

        # ルートノードを初期説明で生成
        root_explanation = self._generator(context)
        root = MCTSNode(explanation=root_explanation)
        root.q_value = self._evaluator_score(context, root_explanation)
        root.visits = 1

        all_nodes: list[MCTSNode] = [root]

        for i in range(self.iterations):
            print(f"[MCTS] イテレーション {i + 1}/{self.iterations}")

            # 1. Select: UCB1 で探索ノードを選択
            selected = self._select(root)

            # 2. Critic: 批判フィードバックを生成
            feedback = self._critic(context, selected.explanation)
            selected.critic_feedback = feedback

            # 3. Refine: 批判を受けて改善した説明を子ノードとして生成
            refined_text = self._refiner(context, selected.explanation, feedback)
            child = MCTSNode(explanation=refined_text, parent=selected)
            selected.children.append(child)
            all_nodes.append(child)

            # 4. Evaluate: Q 値を付与してバックプロパゲーション
            score = self._evaluator_score(context, refined_text)
            self._backpropagate(child, score)

        # 最高 Q 値ノードを選択 (visits>0 かつ平均Q値が最大)
        best = max(
            all_nodes,
            key=lambda n: (n.q_value / n.visits) if n.visits > 0 else 0.0,
        )

        tree_summary = [
            {
                "explanation_snippet": n.explanation[:80] + "...",
                "q_value": round(n.q_value / n.visits, 3) if n.visits > 0 else 0.0,
                "visits": n.visits,
            }
            for n in all_nodes
        ]

        return {
            "step": step,
            "explanation": best.explanation,
            "best_q": round(best.q_value / best.visits, 3) if best.visits > 0 else 0.0,
            "iterations": self.iterations,
            "tree_summary": tree_summary,
            "context": context,
        }

    # ------------------------------------------------------------------
    # MCTS の4役
    # ------------------------------------------------------------------

    def _generator(self, context: StepContext) -> str:
        """Generator (A): 初期説明 (CoT) を生成する。

        Returns:
            説明テキスト
        """
        system = (
            "あなたは空中戦シミュレーターの行動分析専門家です。\n"
            "強化学習エージェントが特定のステップで行った操舵入力の理由を、"
            "センサーデータに基づいて論理的に説明してください。\n"
            f"{_STATE_DESCRIPTION}"
        )
        user = (
            f"【分析対象ステップ: Step {context['step']}】\n\n"
            f"【センサー状態】\n{self._format_state(context['state'])}\n\n"
            f"【操舵入力】\n{self._format_action(context['action'])}\n\n"
            "この操舵入力をエージェントが選択した理由を、センサーデータの数値に基づいて説明してください。"
            "Chain-of-Thought 形式で論理的に推論し、200〜400字程度でまとめてください。"
        )
        return self.llm.simple_prompt(system, user)

    def _critic(self, context: StepContext, explanation: str) -> str:
        """Critic (C): 物理的・論理的矛盾を指摘する批判テキストを返す。

        Returns:
            批判テキスト
        """
        system = (
            "あなたは空中戦シミュレーターの物理法則と戦術の検証専門家です。\n"
            "提示された行動説明に対し、以下の観点から矛盾や不正確な点を批判してください。\n"
            "1. 物理法則との矛盾（例：エレベータを引いたのに高度が下がるとの説明）\n"
            "2. センサー数値との不整合（例：距離が5000mなのに「至近距離」と説明）\n"
            "3. 論理的な飛躍や根拠の欠如\n"
            "批判のない場合は「矛盾なし」と出力してください。\n"
            f"{_STATE_DESCRIPTION}"
        )
        user = (
            f"【センサー状態】\n{self._format_state(context['state'])}\n\n"
            f"【操舵入力】\n{self._format_action(context['action'])}\n\n"
            f"【評価対象の説明】\n{explanation}\n\n"
            "この説明の問題点を批判してください。"
        )
        return self.llm.simple_prompt(system, user)

    def _refiner(
        self,
        context: StepContext,
        explanation: str,
        critic_feedback: str,
    ) -> str:
        """Refiner (A): 批判を受けて改善した説明テキストを返す。

        Returns:
            改善後説明テキスト
        """
        system = (
            "あなたは空中戦シミュレーターの行動分析専門家です。\n"
            "批評家からの指摘を受けて、元の説明の論理的矛盾を修正・改善してください。\n"
            f"{_STATE_DESCRIPTION}"
        )
        user = (
            f"【センサー状態】\n{self._format_state(context['state'])}\n\n"
            f"【操舵入力】\n{self._format_action(context['action'])}\n\n"
            f"【元の説明】\n{explanation}\n\n"
            f"【批評家の指摘】\n{critic_feedback}\n\n"
            "指摘を踏まえて説明を修正・改善してください。センサーの数値を具体的に引用し、"
            "200〜400字程度でまとめてください。"
        )
        return self.llm.simple_prompt(system, user)

    def _evaluator_score(self, context: StepContext, explanation: str) -> float:
        """Evaluator (E): 説明の品質スコアを返す (0.0〜4.0)。

        evaluator.py と同じ Soundness/Fidelity 基準を使う。

        Returns:
            float (soundness + fidelity の合計, 各 0〜2)
        """
        system = (
            "あなたは行動説明の品質を採点する評価専門家です。\n"
            "以下の2つの指標で説明を採点し、JSON形式で出力してください。\n\n"
            "【評価指標（各0〜2の整数）】\n"
            "- soundness: 物理法則・エージェント目標との整合性\n"
            "  0=明らかな矛盾あり, 1=一部不正確, 2=完全に妥当\n"
            "- fidelity: センサー数値との因果関係の正確さ\n"
            "  0=数値を無視/捏造, 1=部分的に正確, 2=完全に正確\n\n"
            '出力形式: {"soundness": int, "fidelity": int, "reason": str}'
        )
        user = (
            f"【センサー状態】\n{self._format_state(context['state'])}\n\n"
            f"【操舵入力】\n{self._format_action(context['action'])}\n\n"
            f"【評価対象の説明】\n{explanation}\n\n"
            "上記の説明を採点してください。"
        )
        raw = self.llm.simple_prompt(
            system, user, response_format={"type": "json_object"}
        )
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError(f"Expected dict, got {type(data)}")
            return float(data.get("soundness", 0)) + float(data.get("fidelity", 0))
        except (json.JSONDecodeError, ValueError, AttributeError):
            # JSON 抽出をフォールバック
            m = re.search(r'"soundness"\s*:\s*(\d)', raw)
            f = re.search(r'"fidelity"\s*:\s*(\d)', raw)
            s = int(m.group(1)) if m else 0
            fi = int(f.group(1)) if f else 0
            return float(s + fi)

    # ------------------------------------------------------------------
    # MCTS 補助
    # ------------------------------------------------------------------

    def _select(self, root: MCTSNode) -> MCTSNode:
        """UCB1 基準でリーフノードを選択する。

        葉ノードでない場合は UCB1 スコア最大の子を再帰的に選択する。

        Returns:
            選択されたリーフノード
        """
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.ucb1(node.visits))
        return node

    def _backpropagate(self, node: MCTSNode, score: float) -> None:
        """スコアをルートまで伝播して visits と q_value を更新する。"""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.q_value += score
            current = current.parent

    # ------------------------------------------------------------------
    # フォーマットヘルパー
    # ------------------------------------------------------------------

    @staticmethod
    def _format_state(state: dict) -> str:
        return "\n".join(f"  {k}: {v:.2f}" for k, v in state.items())

    @staticmethod
    def _format_action(action: dict) -> str:
        return "\n".join(f"  {k}: {v:.2f}" for k, v in action.items())
