"""共通LLMクライアント。

litellm を薄くラップし、環境変数またはコンストラクタ引数でモデルを切り替えられる。
全モジュールはこのクラスだけに依存させることで、モデル変更やテスト時のモック差し替えを容易にする。

設定は .env ファイルまたは環境変数で行う。
  XRL_MODEL_NAME  : litellm モデル名 (例: openai/Qwen3.5-9B, gpt-4o)
  XRL_API_BASE    : エンドポイントURL (例: http://localhost:8088/v1)
  XRL_API_KEY     : APIキー (ローカルサーバーは "dummy" でOK)
  XRL_MAX_TOKENS  : 最大生成トークン数 (デフォルト: 2048)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import litellm

# プロジェクトルートの .env を自動ロード (python-dotenv がなければスキップ)
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)  # 既存の環境変数は上書きしない
    except ImportError:
        pass

_load_dotenv()


class LLMClient:
    """litellm を薄くラップした共通クライアント。

    環境変数 (.env または OS 環境変数):
        XRL_MODEL_NAME: 使用するモデル名 (デフォルト: gpt-4o)
        XRL_API_BASE:   エンドポイントURL (ローカルサーバー用)
        XRL_API_KEY:    APIキー
        XRL_MAX_TOKENS: 最大生成トークン数 (デフォルト: 2048)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.model = model or os.environ.get("XRL_MODEL_NAME", "gpt-4o")
        self.api_key = api_key or os.environ.get("XRL_API_KEY")
        self.api_base = api_base or os.environ.get("XRL_API_BASE")
        self.temperature = temperature
        self.max_tokens = max_tokens or int(os.environ.get("XRL_MAX_TOKENS", "2048"))

    # ------------------------------------------------------------------
    # 公開API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """メッセージ列を送信してテキストを返す。

        Args:
            messages:        OpenAI 形式のメッセージリスト
            temperature:     呼び出し単位で上書き可能 (省略時はインスタンス値)
            max_tokens:      呼び出し単位で上書き可能 (省略時はインスタンス値)
            response_format: {"type": "json_object"} 等のJSON強制オプション

        Returns:
            LLM の返答テキスト

        Raises:
            litellm.exceptions.APIError: API 呼び出し失敗時
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if response_format:
            kwargs["response_format"] = response_format

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content or ""

    def simple_prompt(
        self,
        system: str,
        user: str,
        **kwargs,
    ) -> str:
        """system + user の2行で呼ぶ簡易ラッパー。

        Returns:
            LLM の返答テキスト
        """
        return self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
