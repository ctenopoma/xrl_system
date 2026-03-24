# XRL System — LoRA 学習パイプライン アーキテクチャ

## 概要

XRL (Explainable RL) の説明生成品質を向上させるために、以下の流れで LoRA ファインチューニングを行う。

```
[軌跡ログ CSV]
      │
      ▼
 DatasetBuilder          ← XRL手法 (CoT/MCTS/SySLLM) でラベル生成
      │  dataset/<id>/
      ▼
  LoRA Trainer           ← base model + LoRA rank/lr/epoch を設定
      │  models/<run_id>/adapter/
      ▼
 InferenceEngine         ← ExternalAPI | LocalLoRA (共通 API)
      │
      ▼
 EpisodeEvaluator        ← Soundness / Fidelity を採点
      │
      ▼
  results/baseline/      ← CSV に記録 (学習前後を同一ファイルに蓄積)
```

---

## レイヤー構成

```
modules/
├── prompt_template.py    ← [共有] feature set + プロンプト生成
├── inference_engine.py   ← [推論] PromptTemplate + LLM バックエンド
├── dataset_builder.py    ← [学習] XRL手法でラベル生成 → JSONL 保存
├── lora_trainer.py       ← [学習] HuggingFace PEFT で LoRA 学習
├── llm_client.py         ← 外部 API ラッパー (litellm)
├── evaluator.py          ← LLM-as-a-Judge 採点
├── data_loader.py        ← CSV 読み込み・前処理
└── (既存)
    ├── mcts_xrl.py
    ├── sysllm.py
    └── talktoagent.py

evaluate_baseline.py      ← CLI: 学習なし / 学習後 の評価実行
```

---

## 各レイヤーの詳細

### 1. PromptTemplate (`modules/prompt_template.py`) — 学習・推論で共有

**役割**: feature set とプロンプト文字列を管理する。学習データ生成と推論の両方が
このクラスを通じてフォーマットするため、学習と推論のプロンプトが常に一致する。

```python
from modules.prompt_template import PromptTemplate

# プリセットから生成
tpl = PromptTemplate.from_preset("v1_basic")

# StepContext → (system, user) に変換
system, user = tpl.format_step(context)

# 事前情報スロット付き (v2_with_prior のみ有効)
system, user = tpl.format_step(context, prior_info="エピソード要約: ...")
```

#### プリセット一覧

| preset id        | state features                          | prior_info スロット | 用途             |
|------------------|-----------------------------------------|---------------------|-----------------|
| `v1_basic`       | altitude, speed, distance, ata, aspect  | なし                | 標準ベースライン  |
| `v1_combat_only` | distance, ata, aspect_angle             | なし                | 特徴量削減実験    |
| `v2_with_prior`  | altitude, speed, distance, ata, aspect  | あり                | SySLLM 要約注入  |

#### PromptTemplateConfig フィールド

| フィールド                | デフォルト  | 説明                                   |
|-------------------------|------------|----------------------------------------|
| `template_id`           | "v1_basic" | バージョン識別子 (CSV に記録)             |
| `state_features`        | 全5列       | プロンプトに含めるセンサー列              |
| `action_features`       | 全3列       | プロンプトに含める操舵入力列              |
| `include_sensor_desc`   | True        | センサーの意味説明を system に含めるか    |
| `output_length_hint`    | "200〜400字" | 出力文字数の指示                        |
| `prior_info_slot_enabled` | False     | 事前情報スロットを有効にするか            |

---

### 2. InferenceEngine (`modules/inference_engine.py`) — 推論レイヤー

**役割**: PromptTemplate + LLM バックエンドを統合し、`generate()` の単一 API を提供する。
バックエンドを外部 API / ローカル LoRA で切り替えても呼び出し側は変わらない。

```python
from modules.inference_engine import InferenceEngine, PromptingStrategy

# 外部 API (学習なし baseline)
engine = InferenceEngine(llm, tpl, strategy=PromptingStrategy.ZERO_SHOT)
explanation = engine.generate(context)

# CoT 指示付き
engine_cot = InferenceEngine(llm, tpl, strategy=PromptingStrategy.COT)
explanation = engine_cot.generate(context)

# ローカル LoRA (学習後) ← Step 4 で実装
from modules.inference_engine import LocalLoRABackend
backend = LocalLoRABackend(adapter_path="models/run001/adapter")
engine_lora = InferenceEngine(backend, tpl)
explanation = engine_lora.generate(context)
```

#### PromptingStrategy

| 値           | 動作                                                      |
|-------------|-----------------------------------------------------------|
| `zero_shot` | テンプレートそのまま送信                                    |
| `cot`       | "Chain-of-Thought で推論してから結論を…" をユーザー入力に追加 |

---

### 3. DatasetBuilder (`modules/dataset_builder.py`) — 学習データ生成 ← Step 2

**役割**: XRL 手法を「ラベルソース」として使い、PromptTemplate でフォーマットした
(input, output) ペアを生成して JSONL に保存する。

```python
from modules.dataset_builder import DatasetBuilder, LabelSourceConfig

# ラベルソース設定
label_cfg = LabelSourceConfig(
    method="mcts",         # "cot" | "mcts" | "sysllm"
    min_score=3.0,         # Soundness+Fidelity の最低スコア (品質フィルタ)
    mcts_iterations=4,
)

builder = DatasetBuilder(llm, loader, template=tpl, label_config=label_cfg)
dataset_id = builder.build(steps=range(1, 201))
# → datasets/<dataset_id>/train.jsonl, val.jsonl, metadata.json
```

#### 出力ディレクトリ構成

```
datasets/
└── mcts_v1basic_score3_20260324/
    ├── metadata.json     # 設定・統計の記録
    ├── train.jsonl       # 学習用 (80%)
    └── val.jsonl         # 検証用 (20%)
```

#### JSONL フォーマット (1行 = 1サンプル)

```json
{
  "step": 150,
  "instruction": "(system prompt)",
  "input": "(user prompt: センサー状態 + 操舵入力)",
  "output": "(XRL手法が生成した説明テキスト)",
  "score": 3.5,
  "template_id": "v1_basic",
  "label_source": "mcts"
}
```

#### ラベルソース設定フィールド

| フィールド         | 説明                                              |
|------------------|--------------------------------------------------|
| `method`         | `"cot"`, `"mcts"`, `"sysllm"` のいずれか           |
| `min_score`      | 採点後に残すスコア閾値 (0〜4)。0 はフィルタなし      |
| `mcts_iterations`| method=mcts 時のイテレーション数                   |

---

### 4. LoRA Trainer (`modules/lora_trainer.py`) — 学習 ← Step 3

**役割**: DatasetBuilder が生成した JSONL を使って base model を LoRA でファインチューニングする。

```python
from modules.lora_trainer import LoRATrainer, LoRAConfig

cfg = LoRAConfig(
    base_model="unsloth/Qwen2.5-7B-Instruct",
    lora_rank=16,
    lora_alpha=32,
    learning_rate=2e-4,
    num_epochs=3,
    batch_size=4,
)
trainer = LoRATrainer(cfg)
run_id = trainer.train(dataset_dir="datasets/mcts_v1basic_score3_20260324")
# → models/<run_id>/adapter/ に保存
```

#### LoRAConfig フィールド

| フィールド      | デフォルト                        | 説明                          |
|--------------|----------------------------------|-------------------------------|
| `base_model` | "unsloth/Qwen2.5-7B-Instruct"   | HuggingFace モデル ID          |
| `lora_rank`  | 16                               | LoRA の rank (r)               |
| `lora_alpha` | 32                               | LoRA の alpha                  |
| `learning_rate` | 2e-4                          | 学習率                        |
| `num_epochs` | 3                                | エポック数                     |
| `batch_size` | 4                                | バッチサイズ                   |

#### 出力ディレクトリ構成

```
models/
└── run_20260324_001/
    ├── adapter/          # LoRA アダプタ (safetensors)
    └── run_config.json   # 学習設定の記録 (再現性のため)
```

---

### 5. LocalLoRA Backend — 学習済みモデル推論 ← Step 4

**役割**: `models/<run_id>/adapter/` を読み込んで InferenceEngine に統合する。
外部 API 版と同じ `generate()` インターフェースで動くため、evaluate_baseline.py をそのまま再利用できる。

```python
from modules.inference_engine import LocalLoRABackend, InferenceEngine

backend = LocalLoRABackend(
    adapter_path="models/run_20260324_001/adapter",
    base_model="unsloth/Qwen2.5-7B-Instruct",
)
engine = InferenceEngine(backend, tpl, strategy=PromptingStrategy.ZERO_SHOT)
explanation = engine.generate(context)
```

---

### 6. evaluate_baseline.py — 評価実行 (学習前後共通)

**役割**: 設定を変えながら評価を実行し、`results/baseline/baseline_summary.csv` に追記する。
学習前と学習後で同じスクリプトを使うことで、同一ファイルに比較データが蓄積される。

#### 使い方

```bash
# 学習なし baseline (外部 API)
uv run python evaluate_baseline.py --sample 20

# テンプレートを変えて比較
uv run python evaluate_baseline.py --steps 50 100 150 \
    --template v1_basic v1_combat_only

# CoT 戦略 + SySLLM 事前情報注入
uv run python evaluate_baseline.py --steps 50 100 150 \
    --template v2_with_prior --strategy cot --prior-info sysllm

# 学習後のモデルで評価 ← Step 4 で追加
uv run python evaluate_baseline.py --sample 20 \
    --backend local --adapter models/run_20260324_001/adapter
```

#### 出力ファイル

| ファイル                          | 内容                                      |
|----------------------------------|------------------------------------------|
| `results/baseline/baseline_steps_<ts>.csv` | ステップ単位詳細 (毎回新規)         |
| `results/baseline/baseline_summary.csv`    | 設定ごとサマリー (実行ごとに **追記**) |

##### baseline_steps CSV 列

```
run_at, model, template_id, strategy, prior_info_mode,
step, soundness, fidelity, total, reason
```

##### baseline_summary CSV 列

```
run_at, model, template_id, strategy, prior_info_mode,
n_steps, soundness_mean, soundness_std, fidelity_mean, fidelity_std, total_mean
```

---

## 調整パラメータ一覧

### 学習時の調整項目

| 次元               | 設定場所                  | 例                                   |
|-------------------|--------------------------|--------------------------------------|
| 入力変数 (feature set) | `PromptTemplateConfig` | `v1_basic` / `v1_combat_only`        |
| ラベルソース (XRL手法) | `LabelSourceConfig.method` | `cot` / `mcts` / `sysllm`         |
| ラベル品質フィルタ  | `LabelSourceConfig.min_score` | `0`, `2.0`, `3.0`, `4.0`        |
| LoRA rank         | `LoRAConfig.lora_rank`   | `8`, `16`, `32`                      |
| 学習率             | `LoRAConfig.learning_rate` | `1e-4`, `2e-4`                     |

### 推論時の調整項目

| 次元               | 設定場所                           | 例                                   |
|-------------------|-----------------------------------|--------------------------------------|
| 入力変数           | `PromptTemplateConfig` (学習と同じ) | 学習と必ず一致させること              |
| 事前情報           | `--prior-info`                    | `none` / `sysllm`                    |
| プロンプト戦略     | `--strategy`                      | `zero_shot` / `cot`                  |
| バックエンド       | `--backend`                       | `external` (API) / `local` (LoRA)   |

> **重要**: 入力変数 (`template_id`) は学習と推論で **必ず一致** させること。
> `baseline_summary.csv` の `template_id` 列で確認できる。

---

## 典型的な実験フロー

```bash
# 1. ベースライン評価 (学習なし)
uv run python evaluate_baseline.py --sample 20 \
    --template v1_basic --strategy zero_shot

# 2. 学習データ生成
uv run python build_dataset.py \
    --template v1_basic --label-source mcts --min-score 3.0 --steps all

# 3. LoRA 学習
uv run python train_lora.py \
    --dataset datasets/mcts_v1basic_score3_<ts>/ \
    --rank 16 --epochs 3

# 4. 学習後評価 (同じステップで比較)
uv run python evaluate_baseline.py --sample 20 \
    --template v1_basic --strategy zero_shot \
    --backend local --adapter models/run_<ts>/adapter

# 5. results/baseline/baseline_summary.csv を開いて比較
```

---

## ファイル構成 (完成形)

```
xrl_system/
├── modules/
│   ├── prompt_template.py    ✅ 実装済み
│   ├── inference_engine.py   ✅ 実装済み (外部 API のみ)
│   ├── dataset_builder.py    ✅ 実装済み
│   ├── lora_trainer.py       🔲 Step 3
│   ├── llm_client.py         ✅
│   ├── evaluator.py          ✅
│   ├── data_loader.py        ✅
│   ├── mcts_xrl.py           ✅
│   ├── sysllm.py             ✅
│   └── talktoagent.py        ✅
├── evaluate_baseline.py      ✅ 実装済み
├── build_dataset.py          ✅ 実装済み
├── train_lora.py             🔲 Step 3 (CLI エントリーポイント)
├── datasets/                 (build_dataset.py 実行で生成)
├── models/                   🔲 Step 3 で生成
├── results/baseline/         ✅ evaluate_baseline.py が出力
└── docs/
    ├── architecture.md       ✅ このファイル
    ├── implementation_plan.md ✅
    ├── spec.md               ✅
    ├── usage.md              ✅
    └── development.md        ✅
```
