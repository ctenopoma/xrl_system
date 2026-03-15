# XRL System 使い方ガイド

JSBSim ベースのフライトシミュレーター軌跡ログ (CSV) を LLM で分析・説明する XRL システムの使用方法を説明します。

---

## 目次

1. [セットアップ](#1-セットアップ)
2. [データの準備](#2-データの準備)
3. [実行コマンド一覧](#3-実行コマンド一覧)
4. [各モードの詳細](#4-各モードの詳細)
5. [自動評価オプション](#5-自動評価オプション)
6. [環境変数リファレンス](#6-環境変数リファレンス)
7. [トラブルシューティング](#7-トラブルシューティング)

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

## 4. 各モードの詳細

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

## 5. 自動評価オプション

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

## 6. 環境変数リファレンス

| 環境変数         | デフォルト値 | 説明                                      |
|----------------|-----------|------------------------------------------|
| XRL_MODEL_NAME | gpt-4o    | 分析・評価に使用する LLM モデル名              |
| XRL_API_KEY    | (なし)    | API キー (litellm がサポートする全プロバイダー) |

CLI の `--model` 引数は環境変数より優先されます。

---

## 7. トラブルシューティング

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
