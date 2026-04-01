# XRL System 使い方ガイド

JSBSim ベースのフライトシミュレーター軌跡ログ (CSV) を LLM で分析・説明する XRL システムの使用方法を説明します。

---

## 目次

1. [セットアップ](#1-セットアップ)
2. [データの準備](#2-データの準備)
3. [実行コマンド一覧](#3-実行コマンド一覧)
4. [一括比較モード（compare）](#4-一括比較モードcompare)
5. [各モードの詳細](#5-各モードの詳細)
6. [自動評価オプション](#6-自動評価オプション)
7. [環境変数リファレンス](#7-環境変数リファレンス)
8. [トラブルシューティング](#8-トラブルシューティング)

---

## 1. セットアップ

```bash
# リポジトリのルートで依存パッケージをインストール
pip install -r requirements.txt
```

### 必要な Python バージョン

Python 3.10 以上が必要です（型ヒントに `str | None` 構文を使用しています）。

### API キーの設定

```bash
# OpenAI を使う場合
export XRL_MODEL_NAME="gpt-4o"
export XRL_API_KEY="sk-..."

# Azure OpenAI を使う場合 (litellm 経由)
export XRL_MODEL_NAME="azure/gpt-4o"
export XRL_API_KEY="..."

# Anthropic Claude を使う場合
export XRL_MODEL_NAME="claude-3-5-sonnet-20241022"
export XRL_API_KEY="sk-ant-..."
```

litellm がサポートする全モデルが利用可能です。詳細は [litellm ドキュメント](https://docs.litellm.ai/docs/providers) を参照してください。

---

## 2. データの準備

### CSV ファイルを用意する場合

以下のカラムを持つ CSV ファイルを `data/trajectory_log.csv` に配置してください。

| カラム名      | 型    | 説明                              |
|-------------|-------|-----------------------------------|
| step        | int   | タイムステップ番号                  |
| altitude    | float | 高度 (m)                          |
| speed       | float | 速度 (kt)                         |
| distance    | float | 敵機との距離 (m)                   |
| ata         | float | 自機から敵機へのATA角度 (°)         |
| aspect_angle| float | 敵機から自機への角度 (°)            |
| aileron     | float | エルロン操舵入力 (-1〜1)            |
| elevator    | float | エレベータ操舵入力 (-1〜1)          |
| throttle    | float | スロットル (0〜1)                   |

### CSV なしで動かす場合（ダミーデータ）

`data/trajectory_log.csv` が存在しない場合、システムは自動的にリアルなダミーデータ（200ステップ）を生成して実行します。まず動作確認したい場合はそのまま実行してください。

---

## 3. 実行コマンド一覧

```bash
# モード1: CoT — 特定ステップの行動理由を単一プロンプトで説明
python main.py --method cot --step 150

# モード2: SySLLM — エピソード全体の戦術を要約
python main.py --method sysllm

# モード3: TalkToAgent — 自然言語の質問にデータドリブンで回答
python main.py --method talktoagent --query "Step 150でなぜスロットルを下げたのか検証して"

# モード4: MCTS-XRL — MCTSで反復的に説明を改善
python main.py --method mcts --step 150

# 自動評価付き実行
python main.py --method sysllm --evaluate
python main.py --method mcts --step 150 --evaluate

# オプション指定の例
python main.py --method mcts --step 150 --iterations 6 --model gpt-4-turbo --evaluate
python main.py --method sysllm --csv path/to/my_log.csv --max-rows 30
```

---

## 4. 一括比較モード（compare）

4つの手法を一括実行してスコアを比較する最も簡単な方法です。

```bash
# Step 150 を4手法で比較
python main.py --method compare --step 150

# JSON ファイルに保存
python main.py --method compare --step 150 --output results/compare_step150.json

# 質問文を指定（TalkToAgent に渡される）
python main.py --method compare --step 150 --query "Step 150 で高度を下げた理由を説明して"
```

### 出力される比較テーブル

実行が終わると、全手法のスコアが横並びで表示されます。

```
============================================================
  比較結果サマリー (Step 150)
============================================================
手法           Soundness    Fidelity   合計
------------------------------------------
CoT              2/2          1/2      3/4
SySLLM           2/2          2/2      4/4
MCTS-XRL         2/2          2/2      4/4
TalkToAgent      1/2          1/2      2/4
------------------------------------------
  最高スコア: SySLLM (4/4)
============================================================
```

### 各手法の評価対象

| 手法 | 評価する説明 | `--step` との関係 |
| --- | --- | --- |
| CoT | Step N の行動理由（1ステップ） | 必須。指定ステップを分析 |
| MCTS-XRL | Step N の行動理由（反復改善版） | 必須。指定ステップを分析 |
| TalkToAgent | Step N に関する質問への回答 | 必須。デフォルト質問に使用 |
| SySLLM | エピソード全体の戦術要約 | 無関係。エピソード全体を分析 |

> **注意**: 1回の compare 実行で評価されるのは **1ステップ分** です。複数ステップを評価したい場合は、ステップを変えてコマンドを繰り返し実行してください。

### JSON 出力形式

`--output` を指定した場合、以下の形式で JSON が保存されます。

```json
{
  "step": 150,
  "results": {
    "cot": {
      "step": 150,
      "explanation": "（LLM が生成した説明文）",
      "context": { "step": 150, "state": {...}, "action": {...} },
      "eval": { "soundness": 2, "fidelity": 1, "reason": "..." }
    },
    "sysllm": {
      "tactical_approach": "...",
      "situational_adaptation": "...",
      "inefficiencies": "...",
      "overall_summary": "...",
      "n_keyframes": 12,
      "eval": { "soundness": 2, "fidelity": 2, "reason": "..." }
    },
    "mcts": {
      "step": 150,
      "explanation": "...",
      "best_q": 3.5,
      "iterations": 4,
      "tree_summary": [ { "q_value": 3.5, "visits": 3, "explanation_snippet": "..." } ],
      "eval": { "soundness": 2, "fidelity": 2, "reason": "..." }
    },
    "talktoagent": {
      "query": "Step 150 でどのような戦術的判断をしたか...",
      "plan": "（Coordinator の計画）",
      "code": "（生成された Pandas コード）",
      "exec_output": "（コード実行結果）",
      "explanation": "...",
      "retries": 0,
      "eval": { "soundness": 1, "fidelity": 1, "reason": "..." }
    }
  }
}
```

---

## 5. 各モードの詳細

### モード1: CoT (Chain-of-Thought)

```bash
python main.py --method cot --step <ステップ番号>
```

- 指定したステップのセンサー状態と操舵入力を LLM に渡し、Chain-of-Thought 形式で行動理由を説明します。
- 最もシンプルなモード。速度と精度のバランスが良いです。
- `--step` は必須です。

**出力例:**
```
[Step 150]
状態: {'altitude': 3124.5, 'speed': 248.3, ...}
操舵: {'aileron': -0.82, 'elevator': 0.65, 'throttle': 0.23}

【CoT 説明】
Step 150では、ATAが38.2°と攻撃機会の閾値（45°）を下回っており...
```

### モード2: SySLLM

```bash
python main.py --method sysllm [--max-rows <行数>]
```

- エピソード全体のキーフレーム（攻撃機会・急機動・距離急変）を抽出し、4つの観点で戦術を要約します。
- 長いエピソードでも `--max-rows` でトークン数を制御できます（デフォルト: 50行）。

**出力例:**
```
## 1. 戦術的アプローチ
エージェントは序盤に距離を詰めつつ高度優位を確保する戦術を採用...

## 2. 状況への適応
...
```

### モード3: TalkToAgent

```bash
python main.py --method talktoagent --query "<質問文>"
```

- 4つのエージェント（Coordinator → Coder → Debugger → Explainer）が連携し、Pandas コードを自動生成・実行して回答します。
- `--query` は必須です。
- コード実行は sandbox（許可リストベースのビルトインのみ）で行われます。
- コードエラー時は最大3回自動で修正を試みます。

**質問例:**
```
"Step 100〜200の平均スロットルを計算して"
"ATAが45°未満になった回数を教えて"
"エルロンの絶対値が最大だったステップはどこ？"
```

### モード4: MCTS-XRL

```bash
python main.py --method mcts --step <ステップ番号> [--iterations <回数>]
```

- Generator (初期生成) → Critic (批判) → Refiner (改善) → Evaluator (採点) のサイクルを `--iterations` 回繰り返します。
- 各説明に Q 値（Soundness + Fidelity = 最大4点）を付与し、最高スコアの説明を返します。
- デフォルトのイテレーション回数は4回。精度を上げたい場合は6〜8回に増やしてください（API コスト増加に注意）。

---

## 6. 自動評価オプション

`--evaluate` フラグを付けると、生成された説明を LLM-as-a-Judge（別の LLM インスタンス）が自動採点します。

```bash
python main.py --method cot --step 150 --evaluate
```

**評価指標:**

| 指標          | 説明                                         | スコア範囲 |
|-------------|---------------------------------------------|---------|
| Soundness   | 物理法則・エージェント目標との論理的整合性        | 0〜2    |
| Fidelity    | 実際のセンサー数値との因果関係の正確さ            | 0〜2    |
| 合計         |                                              | 0〜4    |

**出力例:**
```
【評価結果】
  Soundness (論理的妥当性): 2/2
  Fidelity  (忠実性):       1/2
  合計スコア:               3/4
  理由: ATAの数値引用は正確だが、スロットル低下の因果関係が曖昧...
```

---

## 7. 環境変数リファレンス

| 環境変数         | デフォルト値 | 説明                                      |
|----------------|-----------|------------------------------------------|
| XRL_MODEL_NAME | gpt-4o    | 分析・評価に使用する LLM モデル名              |
| XRL_API_KEY    | (なし)    | API キー (litellm がサポートする全プロバイダー) |

CLI の `--model` 引数は環境変数より優先されます。

---

## 8. トラブルシューティング

### API エラーが発生する

- 環境変数 `XRL_API_KEY` が正しく設定されているか確認してください。
- `XRL_MODEL_NAME` が正しいモデル名か確認してください（例: `gpt-4o` は小文字のハイフン区切り）。

### `Step X は存在しません` エラー

- `--step` に指定したステップ番号が CSV に含まれているか確認してください。
- ダミーデータは Step 1〜200 の範囲です。

### TalkToAgent でコードエラーが繰り返し発生する

- 質問を具体的にしてください（例：「step列が何であるか」を含む質問など）。
- より高性能なモデル（例: `gpt-4o` → `gpt-4-turbo`）を試してください。

### SySLLM の出力が短い

- `--max-rows` を小さくしてトークン数を減らすか、逆に大きくして情報量を増やしてみてください。
- LLM の `max_tokens` 設定（`LLMClient` のデフォルト 2048）を変更することも可能です。

---

## 9. evaluate_baseline.py — 学習なしベースライン評価

`evaluate_baseline.py` は複数の XRL 手法・プロンプトテンプレートを **同一ステップで評価・比較** し、スコアと説明文を CSV に記録するスクリプトです。

### 基本的な使い方

```bash
# デフォルト設定 (v1_basic × zero_shot) で 10 ステップをランダム評価
uv run python evaluate_baseline.py

# ステップを指定して評価
uv run python evaluate_baseline.py --steps 50 100 150

# ランダムサンプリング数を指定
uv run python evaluate_baseline.py --sample 20
```

---

### 9-1. XRL 手法を一括比較する（`--method`）

`--method` に手法名を並べると、**同じステップを全手法で評価**して結果を一つの CSV にまとめます。

```bash
# 全5手法を一括比較
uv run python evaluate_baseline.py --sample 10 \
    --method zero_shot cot mcts sysllm agent

# 任意の組み合わせだけ比較
uv run python evaluate_baseline.py --steps 50 100 150 \
    --method zero_shot cot mcts
```

| 手法 | 動作 |
| --- | --- |
| `zero_shot` | `v1_basic` テンプレートをそのまま送信 |
| `cot` | `v1_basic` + "Chain-of-Thought で推論してから…" を追記 |
| `mcts` | Generator→Critic→Refiner×4回で自己改善した最良説明を返す |
| `sysllm` | エピソード要約を生成し `v2_with_prior` テンプレートに注入 |
| `agent` | TalkToAgent が Pandas コードを自動生成・実行して説明 |

> **コスト注意**: `mcts` は 1 ステップあたり LLM を 4〜5 回呼び出します。`agent` も Pandas 実行ループがあります。最初は `--sample 5` 程度で試してください。

---

### 9-2. プロンプトテンプレートを変えながら比較する（`--template`）

`--method` を使わない従来モードでは、`--template` でプリセットを複数指定して比較できます。

```bash
# 3テンプレートを同ステップで比較
uv run python evaluate_baseline.py --steps 50 100 150 \
    --template v1_basic v1_combat_only v2_with_prior

# CoT 戦略で比較
uv run python evaluate_baseline.py --steps 50 100 150 \
    --template v1_basic v1_combat_only --strategy cot

# SySLLM 要約を事前情報として注入
uv run python evaluate_baseline.py --steps 50 100 150 \
    --template v2_with_prior --prior-info sysllm
```

#### 既存プリセット一覧

| preset id | 特徴量 | センサー説明 | 事前情報スロット |
| --- | --- | --- | --- |
| `v1_basic` | 全5列 + 操舵3列 | あり | なし |
| `v1_combat_only` | distance, ata, aspect_angle のみ | あり | なし |
| `v2_with_prior` | 全5列 + 操舵3列 | あり | あり（SySLLM 要約などを注入可能） |

---

### 9-3. プロンプトを変更する方法

プロンプトの編集場所は変えたい内容によって異なります。すべて `modules/prompt_template.py` に集約されています。

#### A. 指示文・質問文を変えたい（最も頻繁）

`_build_user()` メソッド末尾の文字列を編集します。

```python
# 変更前（modules/prompt_template.py 206〜209行あたり）
"この操舵入力をエージェントが選択した理由を、センサーデータの数値に基づいて説明してください。"
f"{self.config.output_length_hint}程度でまとめてください。"

# 例：出力形式を指定する
"この操舵入力を選択した理由を以下の形式で説明してください:\n"
"1. 脅威状況の評価\n2. 選択した行動とその根拠\n3. 期待される効果"
```

#### B. system プロンプトのロール・前提を変えたい

`_build_system()` メソッドの文字列を編集します。

```python
# 変更前（modules/prompt_template.py 186〜190行あたり）
"あなたは空中戦シミュレーターの行動分析専門家です。\n"
"強化学習エージェントが特定のステップで行った操舵入力の理由を、"
"センサーデータに基づいて論理的に説明してください。\n"
```

#### C. 新しいバリエーションをプリセットとして追加したい（推奨）

`PRESETS` 辞書に追記するだけで `--template` から使えます。

```python
# modules/prompt_template.py の PRESETS に追記
PRESETS["v3_concise"] = PromptTemplateConfig(
    template_id="v3_concise",
    state_features=list(STATE_COLS),
    action_features=list(ACTION_COLS),
    include_sensor_desc=False,    # センサー説明なし
    output_length_hint="100字以内",
)
```

```bash
# 追加後すぐに使える
uv run python evaluate_baseline.py --steps 50 100 150 \
    --template v1_basic v3_concise
```

#### D. 出力文字数の指示だけ変えたい

`PRESETS` の `output_length_hint` を書き換えるだけです。

```python
"v1_basic": PromptTemplateConfig(
    ...
    output_length_hint="50〜100字",   # ここだけ変更
),
```

---

### 9-4. 出力ファイル

| ファイル | 内容 |
| --- | --- |
| `results/baseline/baseline_steps_<ts>.csv` | ステップ単位の詳細（毎回新規） |
| `results/baseline/baseline_summary.csv` | 設定ごとのサマリー（実行ごとに追記） |

#### baseline_steps CSV の列

```text
run_at, model, backend, method, template_id, strategy, prior_info_mode,
step, soundness, fidelity, total, explanation, reason
```

- `method` : `--method` モード時は手法名、従来モード時は `{template_id}/{strategy}`
- `explanation` : LLM が生成した説明テキスト本文
- `reason` : Judge（採点 LLM）の採点理由

#### baseline_summary CSV の列

```text
run_at, model, backend, method, template_id, strategy, prior_info_mode,
n_steps, soundness_mean, soundness_std, fidelity_mean, fidelity_std, total_mean
```

---

### 9-5. 典型的な実験フロー

```bash
# Step 1. 手法比較（まず少ないサンプルで確認）
uv run python evaluate_baseline.py --sample 5 \
    --method zero_shot cot mcts

# Step 2. 良さそうな手法に絞ってプロンプトを調整
#   → modules/prompt_template.py の PRESETS に新バリアントを追加

# Step 3. プロンプトバリアントを比較
uv run python evaluate_baseline.py --sample 20 \
    --template v1_basic v3_concise --strategy zero_shot

# Step 4. baseline_summary.csv を開いて method/template_id 列で比較
```
