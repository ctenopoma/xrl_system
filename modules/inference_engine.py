"""InferenceEngine - PromptTemplate + LLMClient / LocalLoRABackend を統合した推論レイヤー。

学習なし (外部 API) と 学習あり (ローカル LoRA) の両方に対応する。
どちらのバックエンドも simple_prompt(system, user) → str インターフェースを持つ。

対応戦略:
    zero_shot : テンプレートをそのまま使用
    cot       : Chain-of-Thought サフィックスを付加
    mcts      : MCTSXRL による Generator/Critic/Refiner/Evaluator の反復自己改善
    sysllm    : SySLLM によるエピソード全体要約を事前情報として注入し説明生成
    agent     : TalkToAgent によるマルチエージェント (Coordinator/Coder/Debugger/Explainer) 推論
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from modules.data_loader import DataLoader, StepContext
from modules.llm_client import LLMClient
from modules.prompt_template import PromptTemplate


# ------------------------------------------------------------------
# プロンプト戦略
# ------------------------------------------------------------------

class PromptingStrategy(str, Enum):
    """推論時のプロンプト戦略。

    ZERO_SHOT : テンプレートそのまま
    COT       : Chain-of-Thought サフィックスを追加
    MCTS      : MCTSXRL による反復自己改善 (Generator/Critic/Refiner/Evaluator)
    SYSLLM    : エピソード全体要約を事前情報として注入して説明生成
    AGENT     : TalkToAgent マルチエージェント (Coordinator/Coder/Debugger/Explainer)
    """
    ZERO_SHOT = "zero_shot"
    COT       = "cot"
    MCTS      = "mcts"
    SYSLLM    = "sysllm"
    AGENT     = "agent"


# ------------------------------------------------------------------
# InferenceEngine
# ------------------------------------------------------------------

class InferenceEngine:
    """PromptTemplate + バックエンド (LLMClient or LocalLoRABackend) を組み合わせた推論エンジン。

    使い方::

        # 外部 API (学習なし baseline)
        tpl    = PromptTemplate.from_preset("v1_basic")
        engine = InferenceEngine(llm, tpl, strategy=PromptingStrategy.ZERO_SHOT)
        explanation = engine.generate(context)

        # ローカル LoRA (学習後)
        from modules.inference_engine import LocalLoRABackend
        backend = LocalLoRABackend(adapter_path="models/run_xxx/adapter")
        engine  = InferenceEngine(backend, tpl)
        explanation = engine.generate(context)

        # MCTS / SySLLM / Agent 戦略 (LLMClient + DataLoader が必要)
        engine = InferenceEngine(llm, tpl,
                                 strategy=PromptingStrategy.MCTS,
                                 loader=loader,
                                 mcts_iterations=4)
        explanation = engine.generate(context)
    """

    def __init__(
        self,
        llm: Union[LLMClient, "LocalLoRABackend"],
        template: PromptTemplate,
        strategy: PromptingStrategy = PromptingStrategy.ZERO_SHOT,
        loader: Optional[DataLoader] = None,
        mcts_iterations: int = 4,
    ) -> None:
        """
        Args:
            llm:             推論バックエンド (LLMClient or LocalLoRABackend)
            template:        プロンプトテンプレート
            strategy:        プロンプト戦略
            loader:          DataLoader (mcts / sysllm / agent 戦略で必要)
            mcts_iterations: MCTS 戦略のイテレーション数 (デフォルト: 4)
        """
        self.llm = llm
        self.template = template
        self.strategy = strategy
        self.loader = loader
        self.mcts_iterations = mcts_iterations

        _requires_loader = {PromptingStrategy.MCTS, PromptingStrategy.SYSLLM, PromptingStrategy.AGENT}
        if strategy in _requires_loader and loader is None:
            raise ValueError(f"strategy='{strategy.value}' には loader が必要です。")

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
            prior_info: 事前情報 (zero_shot / cot 戦略かつ
                        template.config.prior_info_slot_enabled=True の時のみ有効)

        Returns:
            生成された説明テキスト
        """
        if self.strategy == PromptingStrategy.MCTS:
            return self._generate_mcts(context)

        if self.strategy == PromptingStrategy.SYSLLM:
            return self._generate_sysllm(context)

        if self.strategy == PromptingStrategy.AGENT:
            return self._generate_agent(context)

        # zero_shot / cot
        system, user = self.template.format_step(context, prior_info)
        if self.strategy == PromptingStrategy.COT:
            user += self.template.build_cot_suffix()
        return self.llm.simple_prompt(system, user)

    def to_dict(self) -> dict:
        """設定を dict で返す (結果 JSON への記録用)。"""
        return {
            "template": self.template.to_dict(),
            "strategy": self.strategy.value,
            "model": getattr(self.llm, "model", "unknown"),
        }

    # ------------------------------------------------------------------
    # 戦略別プライベートメソッド
    # ------------------------------------------------------------------

    def _generate_mcts(self, context: StepContext) -> str:
        """MCTSXRL による反復自己改善で説明を生成する。"""
        from modules.mcts_xrl import MCTSXRL

        mcts = MCTSXRL(self.llm, self.loader, iterations=self.mcts_iterations)
        result = mcts.explain_mcts(context.step)
        return result["explanation"]

    def _generate_sysllm(self, context: StepContext) -> str:
        """エピソード全体要約を事前情報として注入し説明を生成する。

        SySLLM でエピソード全体を要約したうえで、その要約を prior_info として
        v2_with_prior テンプレートに注入して per-step の説明を生成する。
        テンプレートが prior_info_slot_enabled でない場合はサフィックスとして付加する。
        """
        from modules.sysllm import SySLLM

        sys_result = SySLLM(self.llm, self.loader).analyze()
        summary = sys_result.get("overall_summary", "")

        if self.template.config.prior_info_slot_enabled:
            system, user = self.template.format_step(context, prior_info=summary)
        else:
            system, user = self.template.format_step(context)
            user += f"\n\n【エピソード全体要約】\n{summary}"

        return self.llm.simple_prompt(system, user)

    def _generate_agent(self, context: StepContext) -> str:
        """TalkToAgent マルチエージェントで説明を生成する。"""
        from modules.talktoagent import TalkToAgent

        query = (
            f"Step {context.step} において、エージェントはなぜこの行動を選択したのか、"
            "センサーデータと操舵入力の関係を具体的に説明してください。"
        )
        agent = TalkToAgent(self.llm, self.loader)
        result = agent.answer(query)
        return result["explanation"]


# ------------------------------------------------------------------
# LocalLoRABackend — 学習済みアダプタを使ったローカル推論バックエンド
# ------------------------------------------------------------------

class LocalLoRABackend:
    """学習済み LoRA アダプタを読み込んでローカルで推論するバックエンド。

    LLMClient と同じ simple_prompt(system, user) → str インターフェースを持つ。
    モデルは最初の呼び出し時に遅延ロードされる。

    使い方::

        backend = LocalLoRABackend(
            adapter_path="models/run_xxx/adapter",
        )
        response = backend.simple_prompt(system, user)

        # ベースモデルを明示的に指定する場合
        backend = LocalLoRABackend(
            adapter_path="models/run_xxx/adapter",
            base_model="Qwen/Qwen2.5-7B-Instruct",
        )
    """

    def __init__(
        self,
        adapter_path: str | Path,
        base_model: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        """
        Args:
            adapter_path:    LoRA アダプタのディレクトリ (train_lora.py が生成する adapter/ フォルダ)
            base_model:      ベースモデルの HuggingFace ID またはローカルパス。
                             省略時は adapter_path/../run_config.json から自動読み込み。
            max_new_tokens:  最大生成トークン数
            temperature:     生成温度 (0 で greedy デコード)
        """
        self.adapter_path = Path(adapter_path)
        self._base_model_arg = base_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # LLMClient 互換プロパティ
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """to_dict() 互換: アダプタパスを識別子として返す。"""
        return f"local:{self.adapter_path}"

    # ------------------------------------------------------------------
    # 公開 API (LLMClient と同じインターフェース)
    # ------------------------------------------------------------------

    def simple_prompt(self, system: str, user: str, **_kwargs) -> str:
        """system + user を受け取り、学習済みモデルの応答を返す。

        Args:
            system: システムプロンプト
            user:   ユーザープロンプト

        Returns:
            生成されたテキスト
        """
        if self._model is None:
            self._load()
        return self._generate(system, user)

    # ------------------------------------------------------------------
    # 非公開ヘルパー
    # ------------------------------------------------------------------

    def _resolve_base_model(self) -> str:
        """ベースモデル名を解決する。"""
        if self._base_model_arg:
            return self._base_model_arg

        # adapter_path/../run_config.json から読む
        config_path = self.adapter_path.parent / "run_config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            base = cfg.get("base_model")
            if base:
                return base

        raise ValueError(
            f"base_model が特定できません。\n"
            f"  --base-model で明示的に指定するか、\n"
            f"  {config_path} に base_model が記録されていることを確認してください。"
        )

    def _load(self) -> None:
        """トークナイザーとモデルを遅延ロードする (最初の呼び出し時に実行)。"""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model = self._resolve_base_model()

        print(f"[LocalLoRABackend] トークナイザーを読み込み中: {self.adapter_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.adapter_path),
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )

        print(f"[LocalLoRABackend] ベースモデルを読み込み中: {base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"[LocalLoRABackend] LoRA アダプタを適用中: {self.adapter_path}")
        self._model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self._model.eval()
        print("[LocalLoRABackend] 読み込み完了")

    def _generate(self, system: str, user: str) -> str:
        """プロンプトをトークナイズして生成し、応答テキストを返す。"""
        import torch

        # chat template でフォーマット (なければフォールバック)
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        try:
            if getattr(self._tokenizer, "chat_template", None):
                prompt_text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt_text = (
                    f"### System:\n{system}\n\n"
                    f"### User:\n{user}\n\n"
                    f"### Assistant:\n"
                )
        except Exception:
            prompt_text = (
                f"### System:\n{system}\n\n"
                f"### User:\n{user}\n\n"
                f"### Assistant:\n"
            )

        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # 入力トークンを除いた応答部分だけをデコード
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
