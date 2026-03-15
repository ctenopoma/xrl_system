"""TalkToAgent アプローチ。

Coordinator → Coder → Debugger → Explainer の4役マルチエージェントが協調し、
Pandas コードを自動生成・実行してデータから対話的に回答を導く。
"""

from __future__ import annotations

import contextlib
import io
import re
import traceback

from modules.data_loader import DataLoader
from modules.llm_client import LLMClient

MAX_DEBUG_RETRIES = 3

# Coder が生成するコードで許可するビルトイン関数
_SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
}


class TalkToAgent:
    """Coordinator/Coder/Debugger/Explainer の4役マルチエージェント。

    使い方::

        agent = TalkToAgent(llm, loader)
        result = agent.answer("Step 150でなぜスロットルを下げたのか検証して")
        print(result["explanation"])
    """

    def __init__(self, llm: LLMClient, loader: DataLoader) -> None:
        self.llm = llm
        self.loader = loader
        self._df = loader.load()

    # ------------------------------------------------------------------
    # 公開エントリーポイント
    # ------------------------------------------------------------------

    def answer(self, query: str) -> dict:
        """ユーザー質問にエンドツーエンドで回答を生成する。

        Args:
            query: ユーザーの自然言語質問

        Returns:
            {
              "query":        str,
              "plan":         str,   # Coordinator 出力
              "code":         str,   # 最終実行コード
              "exec_output":  str,   # コード実行結果
              "explanation":  str,   # Explainer 出力 (最終回答)
              "retries":      int,   # Debugger が再試行した回数
            }
        """
        plan = self._coordinator(query)
        code, exec_output, retries = self._coder_debugger_loop(query, plan)
        explanation = self._explainer(query, exec_output)
        return {
            "query": query,
            "plan": plan,
            "code": code,
            "exec_output": exec_output,
            "explanation": explanation,
            "retries": retries,
        }

    # ------------------------------------------------------------------
    # 各エージェントの実装
    # ------------------------------------------------------------------

    def _coordinator(self, query: str) -> str:
        """質問を解釈し、Pandas 処理計画を自然言語で返す。

        Returns:
            処理計画テキスト
        """
        df_info = self._get_df_info()
        system = (
            "あなたはデータ分析の計画立案エージェント (Coordinator) です。\n"
            "ユーザーの質問と DataFrame の概要を受け取り、"
            "質問に答えるための Pandas 処理ステップを箇条書きで計画してください。\n"
            "コードは書かず、処理手順のみを日本語で記述してください。"
        )
        user = (
            f"【DataFrame の概要】\n{df_info}\n\n"
            f"【ユーザーの質問】\n{query}\n\n"
            "この質問に答えるためのデータ処理計画を立ててください。"
        )
        return self.llm.simple_prompt(system, user)

    def _coder(self, query: str, plan: str, error_log: str = "") -> str:
        """処理計画（とオプションのエラーログ）から Pandas コードを生成する。

        Args:
            error_log: 前回実行時のエラー。初回は空文字列。

        Returns:
            実行可能な Python コードブロック
        """
        df_info = self._get_df_info()
        error_section = (
            f"\n\n【前回のエラーログ】\n{error_log}\n上記エラーを修正してください。"
            if error_log
            else ""
        )
        system = (
            "あなたはデータ分析コードを生成するエージェント (Coder) です。\n"
            "以下の制約に従い、Python (Pandas) コードを生成してください。\n\n"
            "【制約】\n"
            "- 変数 `df` に既にデータが読み込まれています（再読み込み不要）\n"
            "- `pd` (pandas) は利用可能です\n"
            "- `import` 文は使用できません\n"
            "- ファイル読み書き (`open`, `to_csv` 等) は使用できません\n"
            "- 結果は必ず `print()` で出力してください\n"
            "- コードのみを ```python ``` ブロックで囲って出力してください"
        )
        user = (
            f"【DataFrame の概要】\n{df_info}\n\n"
            f"【処理計画】\n{plan}\n\n"
            f"【ユーザーの質問】\n{query}"
            f"{error_section}\n\n"
            "上記の計画を実行する Python コードを生成してください。"
        )
        raw = self.llm.simple_prompt(system, user)
        return self._extract_code_block(raw)

    def _execute_code(self, code: str) -> tuple[str, str | None]:
        """sandboxed なスコープでコードを実行する。

        sandbox 設計:
            - exec() の globals に許可リストのビルトインと pd/df のみを注入
            - stdout を io.StringIO でキャプチャ
            - import やファイル操作は NameError で弾かれる

        Args:
            code: 実行する Python コード文字列

        Returns:
            (stdout_output, error)
            正常終了時 error=None、エラー時 error=トレースバック文字列
        """
        import pandas as pd  # noqa: PLC0415

        safe_globals: dict = {
            "__builtins__": _SAFE_BUILTINS,
            "pd": pd,
            "df": self._df.copy(),  # 破壊的変更を防ぐため copy
        }

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, safe_globals)  # noqa: S102
            return buf.getvalue(), None
        except Exception:
            return "", traceback.format_exc()

    def _coder_debugger_loop(
        self, query: str, plan: str
    ) -> tuple[str, str, int]:
        """Coder → Debugger のリトライループ (最大 MAX_DEBUG_RETRIES 回)。

        Returns:
            (final_code, exec_output, retry_count)

        Raises:
            RuntimeError: MAX_DEBUG_RETRIES 回失敗した場合
        """
        error_log = ""
        for attempt in range(MAX_DEBUG_RETRIES + 1):
            code = self._coder(query, plan, error_log=error_log)
            output, error = self._execute_code(code)
            if error is None:
                return code, output, attempt
            error_log = error
            print(f"[TalkToAgent] コード実行失敗 (試行 {attempt + 1}/{MAX_DEBUG_RETRIES})")

        raise RuntimeError(
            f"コードの修正に {MAX_DEBUG_RETRIES} 回失敗しました。\n最後のエラー:\n{error_log}"
        )

    def _explainer(self, query: str, exec_output: str) -> str:
        """実行結果をユーザー向けの自然言語説明に変換する。

        Returns:
            最終回答テキスト
        """
        system = (
            "あなたはデータ分析結果をわかりやすく説明するエージェント (Explainer) です。\n"
            "コードの実行出力を受け取り、ユーザーの質問に対して日本語で丁寧に答えてください。\n"
            "数値は具体的に引用し、論理的な因果関係を説明してください。"
        )
        user = (
            f"【ユーザーの質問】\n{query}\n\n"
            f"【データ分析の実行結果】\n{exec_output}\n\n"
            "上記の結果を踏まえて、ユーザーの質問に答えてください。"
        )
        return self.llm.simple_prompt(system, user)

    # ------------------------------------------------------------------
    # 非公開ヘルパー
    # ------------------------------------------------------------------

    def _get_df_info(self) -> str:
        """DataFrame のカラム情報と先頭数行を文字列で返す。"""
        cols = list(self._df.columns)
        n = len(self._df)
        step_range = f"{self._df['step'].min()}〜{self._df['step'].max()}"
        return (
            f"カラム: {cols}\n"
            f"行数: {n}\n"
            f"Step 範囲: {step_range}\n"
            f"先頭5行:\n{self._df.head().to_string()}"
        )

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """LLM 応答から ```python ... ``` ブロックのコードを抽出する。

        ブロックが見つからない場合はテキスト全体を返す。
        """
        match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
