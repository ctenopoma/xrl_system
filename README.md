# XRL System

JSBSim ベースのフライトシミュレーター（[LAG](LAG/)）で学習した強化学習エージェントの軌跡ログ（CSV）を LLM で分析・説明する **Explainable RL (XRL)** システムです。

## 概要

4つの分析モードと LLM-as-a-Judge による自動評価を組み合わせ、エージェントの戦術・行動理由を定量的に比較できます。

| モード | コマンド | 説明 |
| ------ | -------- | ---- |
| CoT | `--method cot` | 特定ステップの行動理由を単一プロンプトで説明 |
| SySLLM | `--method sysllm` | エピソード全体の戦術をトップダウンで要約 |
| TalkToAgent | `--method talktoagent` | Pandas コードを自動生成・実行してデータドリブンに回答 |
| MCTS-XRL | `--method mcts` | MCTS で Generator/Critic/Refiner/Evaluator を反復させ説明を自己改善 |

## ディレクトリ構成

```text
xrl_system/
├── main.py                  # CLI エントリーポイント
├── requirements.txt
├── .env.example             # 環境変数テンプレート
├── data/                    # trajectory_log.csv の配置先
├── modules/
│   ├── llm_client.py        # 共通 LLM クライアント (litellm ラッパー)
│   ├── data_loader.py       # CSV 読み込み・ダミー生成・前処理
│   ├── sysllm.py            # モード2: 全体要約
│   ├── talktoagent.py       # モード3: マルチエージェント対話
│   ├── mcts_xrl.py          # モード1 & 4: CoT / MCTS-XRL
│   └── evaluator.py         # LLM-as-a-Judge 自動評価
├── docs/
│   ├── spec.md              # 実装仕様書
│   ├── usage.md             # 使い方ガイド
│   └── development.md       # 改修ガイド
└── LAG/                     # JSBSim 強化学習環境
    └── envs/JSBSim/data/aircraft/f16/
```

## セットアップ

```bash
pip install -r requirements.txt
```

### LLM の設定

`.env.example` をコピーして `.env` を作成します。

```bash
cp .env.example .env
```

**ローカル llama.cpp サーバーの場合（デフォルト設定）:**

```env
XRL_MODEL_NAME=openai/Qwen3.5-9B
XRL_API_BASE=http://localhost:8088/v1
XRL_API_KEY=dummy
XRL_MAX_TOKENS=2048
```

**OpenAI の場合:**

```env
XRL_MODEL_NAME=gpt-4o
XRL_API_KEY=sk-...
```

litellm 経由で Anthropic・Ollama・vLLM 等すべてのプロバイダーが利用可能です。

## データの準備

`data/trajectory_log.csv` に以下のカラムを持つ CSV を配置します。

| カラム | 説明 |
| ------ | ---- |
| step | タイムステップ |
| altitude | 高度 (m) |
| speed | 速度 (kt) |
| distance | 敵機との距離 (m) |
| ata | 自機→敵機 ATA 角度 (°) |
| aspect_angle | 敵機→自機 角度 (°) |
| aileron | エルロン操舵 (-1〜1) |
| elevator | エレベータ操舵 (-1〜1) |
| throttle | スロットル (0〜1) |

CSV がない場合は自動でダミーデータ（200ステップ）が生成されます。

## 実行例

```bash
# エピソード全体の戦術を要約して自動評価
python main.py --method sysllm --evaluate

# Step 150 の行動理由を CoT で説明
python main.py --method cot --step 150 --evaluate

# Step 150 を MCTS で反復改善（8回）して自動評価
python main.py --method mcts --step 150 --iterations 8 --evaluate

# 自然言語の質問にデータドリブンで回答
python main.py --method talktoagent --query "Step 150でなぜスロットルを下げたのか検証して"

# 別の CSV ファイルを指定
python main.py --method sysllm --csv path/to/my_log.csv
```

## 自動評価スコア

`--evaluate` を付けると LLM-as-a-Judge が生成した説明を採点します。

| 指標 | 内容 | スコア |
| ---- | ---- | ------ |
| Soundness | 物理法則・目標との論理的整合性 | 0〜2 |
| Fidelity | センサー数値との因果関係の正確さ | 0〜2 |
| 合計 | | 0〜4 |

## ドキュメント

- [使い方ガイド](docs/usage.md) — コマンドの詳細・トラブルシューティング
- [改修ガイド](docs/development.md) — モジュール別の拡張・変更方法
- [実装仕様書](docs/spec.md) — 設計の詳細仕様
