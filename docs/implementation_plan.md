# 実装計画

## 全体ステップ

| ステップ | 内容                         | 状態   |
|---------|------------------------------|--------|
| Step 1  | PromptTemplate / InferenceEngine / baseline 評価 | ✅ 完了 |
| Step 2  | DatasetBuilder (学習データ生成) | ✅ 完了 |
| Step 3  | LoRA Trainer                  | ✅ 完了 |
| Step 4  | LocalLoRA Backend (学習済み推論) | ✅ 完了 |
| Step 5  | 実験管理 (config sweep / 結果比較) | 🔲 未着手 |

---

## Step 1 — PromptTemplate / InferenceEngine / baseline 評価 ✅

### 実装内容

- `modules/prompt_template.py`
  - `PromptTemplateConfig` データクラス
  - `PromptTemplate` クラス: `format_step(context, prior_info)` → `(system, user)`
  - プリセット: `v1_basic`, `v1_combat_only`, `v2_with_prior`
- `modules/inference_engine.py`
  - `PromptingStrategy` enum: `zero_shot`, `cot`
  - `InferenceEngine(llm, template, strategy)`: `generate(context, prior_info)` → str
- `evaluate_baseline.py`
  - 複数テンプレート × 複数ステップの評価実行
  - `results/baseline/baseline_steps_<ts>.csv` (詳細)
  - `results/baseline/baseline_summary.csv` (追記式サマリー)

### 使い方

```bash
# 標準実行 (10ステップをランダムサンプル)
uv run python evaluate_baseline.py --sample 10

# 複数テンプレートを一括比較
uv run python evaluate_baseline.py --steps 50 100 150 \
    --template v1_basic v1_combat_only

# SySLLM 要約を事前情報として注入
uv run python evaluate_baseline.py --steps 50 100 150 \
    --template v2_with_prior --prior-info sysllm --strategy cot
```

---

## Step 2 — DatasetBuilder (学習データ生成) ✅

### 目的

XRL 手法 (CoT / MCTS / SySLLM) をラベルソースとして使い、
LoRA 学習に使える `(instruction, input, output)` 形式の JSONL データセットを生成する。

### 実装内容

#### `modules/dataset_builder.py`

```python
@dataclass
class LabelSourceConfig:
    method: str           # "cot" | "mcts" | "sysllm"
    min_score: float      # 品質フィルタ (Soundness+Fidelity の最低スコア, 0〜4)
    mcts_iterations: int  # method="mcts" 時のイテレーション数

class DatasetBuilder:
    def build(self, steps: Iterable[int]) -> str:
        """説明を生成・採点し、閾値以上のものを JSONL に保存。dataset_id を返す。"""
```

処理フロー:
1. 指定ステップ全てに対して XRL 手法で説明を生成
2. EpisodeEvaluator でスコアリング (Soundness + Fidelity)
3. `min_score` 未満を除外
4. PromptTemplate で `(instruction, input)` を生成し `output` と対応付け
5. 80/20 で train/val に分割して JSONL 保存
6. `metadata.json` に設定・統計を記録

#### 出力フォーマット

`datasets/<method>_<template>_score<min>_<ts>/`

```
├── train.jsonl
├── val.jsonl
└── metadata.json
```

JSONL 1行:
```json
{
  "instruction": "(system prompt)",
  "input": "(user prompt)",
  "output": "(XRL生成の説明テキスト)",
  "step": 150,
  "score": 3.5,
  "template_id": "v1_basic",
  "label_source": "mcts"
}
```

#### `build_dataset.py` (CLI エントリーポイント)

```bash
# MCTS で全ステップ、スコア3.0以上を収集
uv run python build_dataset.py \
    --template v1_basic \
    --label-source mcts \
    --min-score 3.0 \
    --mcts-iterations 4

# CoT で高速生成 (フィルタなし)
uv run python build_dataset.py \
    --template v1_basic \
    --label-source cot \
    --min-score 0
```

### 確認観点

- [ ] 生成されたサンプル数が十分か (`metadata.json` の `n_train` を確認)
- [ ] `val.jsonl` の score 分布を確認 (品質フィルタの効果測定)
- [ ] `instruction` / `input` が `evaluate_baseline.py` の出力と一致しているか

---

## Step 3 — LoRA Trainer ✅

### 目的

Step 2 で生成したデータセットを使って、ローカル LLM を LoRA でファインチューニングする。

### 実装内容

#### `modules/lora_trainer.py`

```python
@dataclass
class LoRAConfig:
    base_model: str       # HuggingFace モデル ID
    lora_rank: int        # LoRA rank (r)
    lora_alpha: int       # LoRA alpha
    learning_rate: float
    num_epochs: int
    batch_size: int
    max_seq_length: int   # 最大シーケンス長

class LoRATrainer:
    def train(self, dataset_dir: str) -> str:
        """学習を実行し、run_id を返す。アダプタは models/<run_id>/adapter/ に保存。"""
```

使用ライブラリ: `unsloth` (高速 LoRA) + `transformers` + `trl` (SFTTrainer)

#### `train_lora.py` (CLI エントリーポイント)

```bash
# 基本実行
uv run python train_lora.py \
    --dataset datasets/mcts_v1basic_score3_<ts>/ \
    --base-model unsloth/Qwen2.5-7B-Instruct \
    --rank 16 \
    --epochs 3

# rank を変えて実験
uv run python train_lora.py \
    --dataset datasets/mcts_v1basic_score3_<ts>/ \
    --rank 8 --epochs 3
```

#### 出力

```
models/
└── run_20260324_001/
    ├── adapter/          # LoRA アダプタ (safetensors)
    └── run_config.json   # 学習設定の完全な記録
```

`run_config.json` 例:
```json
{
  "run_id": "run_20260324_001",
  "base_model": "unsloth/Qwen2.5-7B-Instruct",
  "dataset_dir": "datasets/mcts_v1basic_score3_20260324/",
  "lora_rank": 16,
  "learning_rate": 0.0002,
  "num_epochs": 3,
  "n_train_samples": 145,
  "final_train_loss": 0.312
}
```

### 確認観点

- [ ] 学習ロスが収束しているか (train_loss の推移)
- [ ] val loss が train loss に比べて大きく乖離していないか (過学習チェック)
- [ ] `models/<run_id>/adapter/` にファイルが生成されているか

---

## Step 4 — LocalLoRA Backend (学習済みモデル推論) ✅

### 目的

Step 3 で学習したアダプタを InferenceEngine に組み込み、
学習後のモデルで評価できるようにする。

### 実装内容

#### `modules/inference_engine.py` への追加

```python
class LocalLoRABackend:
    """unsloth / transformers で LoRA アダプタを読み込むバックエンド。"""

    def __init__(self, adapter_path: str, base_model: str):
        ...

    def simple_prompt(self, system: str, user: str, **kwargs) -> str:
        """LLMClient と同じインターフェースで応答を返す。"""
        ...
```

`InferenceEngine` は `llm` に `LLMClient` か `LocalLoRABackend` かを問わず動く。

#### `evaluate_baseline.py` への追加

```bash
# --backend と --adapter オプションを追加
uv run python evaluate_baseline.py --sample 20 \
    --template v1_basic \
    --backend local \
    --adapter models/run_20260324_001/adapter \
    --base-model unsloth/Qwen2.5-7B-Instruct
```

### 確認観点

- [ ] 学習前 (external) と学習後 (local) で同じ `template_id` を使っているか
- [ ] `baseline_summary.csv` に学習前後の行が並んで記録されているか

---

## Step 5 — 実験管理 (config sweep / 結果比較) 🔲

### 目的

複数の設定を体系的に比較し、どの組み合わせが最も性能向上に寄与するかを把握する。

### 実装内容

#### `run_experiment.py` (複数設定を一括実行)

```bash
# テンプレート × ラベルソース × rank の全組み合わせを実行
uv run python run_experiment.py --config configs/sweep_example.yaml
```

`configs/sweep_example.yaml`:
```yaml
template:      [v1_basic, v1_combat_only]
label_source:  [cot, mcts]
min_score:     [2.0, 3.0]
lora_rank:     [8, 16]
eval_steps:    [50, 100, 150, 180]
eval_strategy: zero_shot
```

処理フロー:
1. 各設定で `build_dataset.py` を実行
2. 各データセットで `train_lora.py` を実行
3. 各アダプタで `evaluate_baseline.py` を実行
4. 全結果が `baseline_summary.csv` に追記される

#### `compare_results.py` (CSV から比較レポートを出力)

```bash
uv run python compare_results.py \
    --summary results/baseline/baseline_summary.csv \
    --output results/comparison_report.md
```

出力例:
```markdown
## 比較結果

| template_id   | label_source | lora_rank | total_mean | vs baseline |
|---------------|-------------|-----------|-----------|-------------|
| (baseline)    | —           | —         | 2.45      | —           |
| v1_basic      | mcts        | 16        | 3.10      | +0.65       |
| v1_combat_only| cot         | 8         | 2.80      | +0.35       |
```

### 確認観点

- [ ] ベースライン (学習なし) との差分が正しく計算されているか
- [ ] 設定の再現に必要な情報が全て `baseline_summary.csv` に記録されているか

---

## 優先度と依存関係

```
Step 1 (✅)
    └── Step 2 (DatasetBuilder) — 学習データがなければ Step 3 は不可
            └── Step 3 (LoRA Trainer) — アダプタがなければ Step 4 は不可
                    └── Step 4 (LocalLoRA Backend)
                            └── Step 5 (実験管理) — Step 1〜4 が揃って初めて有意義な比較が可能
```

Step 2 → Step 3 → Step 4 の順に着手する。Step 5 は Step 4 完了後。
