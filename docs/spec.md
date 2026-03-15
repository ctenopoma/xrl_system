# 強化学習(RL)エージェントの包括的説明性(XRL)分析・比較システム 実装仕様書

## 1. プロジェクト概要

フライトシミュレータ（JSBSimベースのLAG環境など）から出力されたRLエージェントの軌跡ログ（CSV）を読み込み、複数の最先端LLMベース手法を用いてエージェントの戦術ポリシーや行動理由を自然言語で説明（XRL: Explainable RL）するシステムを構築する。
また、生成された説明の品質をLLM自身に評価させる自動評価フレームワーク（LLM-as-a-Judge）を実装し、各手法の有効性を定量的に比較できるようにする。

## 2. 実装する分析アプローチ

本システムでは、以下の4つのモードを実装する。

* **モード1: CoT / SFT (Baseline)**: 単一のプロンプトによる基本的な推論（Chain-of-Thought）。ファインチューニング済みモデル（SFT）の推論にも使用する。
* **モード2: SySLLM**: 間引きされた軌跡全体を入力とし、エージェントの「全体的な戦術・弱点」をトップダウンで一括要約する。
* **モード3: TalkToAgent**: Coordinator, Coder, Debugger, Explainerの複数エージェントが協調し、Pandasコードを自動生成・実行してデータから対話的に回答を導き出す。
* **モード4: MCTS-XRL**: モンテカルロ木探索（MCTS）を用い、LLMが「生成」「批判」「評価」「改善」のロールを担って、特定のステップの行動理由に対する説明を反復的に自己改善する。

## 3. ディレクトリ構成と共通モジュール

```text
xrl_system/
├── data/                  # CSVデータ配置用
├── modules/
│   ├── data_loader.py     # CSV読み込み・ダミー生成・共通フィルタリング
│   ├── sysllm.py          # モード2: 全体要約モジュール
│   ├── talktoagent.py     # モード3: マルチエージェント対話モジュール
│   ├── mcts_xrl.py        # モード1 & 4: CoTおよびMCTSベースの局所説明モジュール
│   └── evaluator.py       # LLM-as-a-Judge 自動評価モジュール
├── requirements.txt       # pandas, litellm などを指定
└── main.py                # CLIエントリーポイント

```

### `data_loader.py` の仕様

* **対象データ**: `trajectory_log.csv` (存在しない場合はモックデータを生成する)。
* **カラム**: `step`, `altitude`, `speed`, `distance`, `ata` (自機から敵への角度), `aspect_angle` (敵機から自機への角度), `aileron`, `elevator`, `throttle`
* **機能**: DataFrameとしての読み込み機能と、SySLLM向けに「攻撃機会(`ata`<45)」「急機動(操舵の0.5以上の変化)」「距離の急変」に基づくフィルタリング＆自然言語テキスト化機能を提供する。

---

## 4. 分析モジュールの詳細仕様

### 4.1. `sysllm.py` (SySLLMアプローチ)

* **目的**: エピソード全体の戦術を要約する。
* **処理**: `data_loader.py` でテキスト化された軌跡履歴をLLMに渡し、「1. 戦術的アプローチ」「2. 状況への適応」「3. 非効率性と弱点」「4. 総合要約」の4項目を一度のAPIコールで生成させる。

### 4.2. `talktoagent.py` (TalkToAgentアプローチ)

* **目的**: ユーザーの質問（例：「なぜStep 150で急降下した？」）にデータドリブンで答える。
* **処理フロー**:
1. **Coordinator**: 質問を解釈し、検証のためのPython(Pandas)処理計画を立てる。
2. **Coder**: 計画に基づき、CSVを読み込んで統計量の計算や特定ステップの抽出を行うPythonコードを出力する。
3. **Debugger**: `exec()` または `subprocess` を用いて安全にコードを実行。エラーがあればCoderにエラーログを渡し修正させる（最大3回）。
4. **Explainer**: コードの実行出力結果を受け取り、ユーザーへの最終的な解説テキストを生成する。



### 4.3. `mcts_xrl.py` (CoT / MCTSアプローチ)

* **目的**: 指定された特定Stepの「行動理由」について、論理的矛盾のない正確な説明を生成する。
* **処理フロー (MCTS)**:
1. **Generator (A)**: 対象ステップの状況・行動から、初期の説明（CoT）を生成する（ルートノード）。
2. **Critic (C)**: 生成された説明に対し、物理法則（例：「エレベータを引いたのに高度が下がると説明している」等）の矛盾がないか批判を行う。
3. **Refiner (A)**: Criticの指摘に基づき、説明を修正して子ノードを生成する。
4. **Evaluator (E)**: 各ノード（説明）に対し、下記の `evaluator.py` と同じ基準でスコア（Q値）を付与する。
5. 探索をN回（例: 4回）繰り返し、最高のQ値を持つノードのテキストを最終結果とする。



---

## 5. `evaluator.py` (LLM-as-a-Judge 評価モジュール)

* **目的**: 各手法で生成された説明文を、別のLLMインスタンス（Judge）が自動採点する。
* **評価指標**（各0, 1, 2の3段階評価）:
* **Soundness (論理的妥当性)**: 環境の物理的メカニズムと論理的に矛盾していないか。エージェントの目的に合致しているか。
* **Fidelity (忠実性)**: 実際の状態数値（距離や角度の変化）に基づいた正しい因果関係が説明されているか。無関係な要素を捏造していないか。


* **出力**: LLMにJSONフォーマットで `{"soundness": int, "fidelity": int, "reason": str}` を出力させ、パースして記録する。

---

## 6. `main.py` (CLIインターフェース)

`argparse` を使用して、実行する手法と対象データを指定できるようにする。

**コマンドライン引数の例**:

* `--method`: 実行する手法 (`cot`, `sysllm`, `talktoagent`, `mcts`) を指定。
* `--step`: `cot` や `mcts` で分析対象とするステップ番号を指定（必須）。
* `--query`: `talktoagent` でのユーザー質問を指定。
* `--evaluate`: このフラグを付けた場合、生成された説明に対して最後に `evaluator.py` を実行し、Soundness/Fidelityスコアをコンソールに出力する。

**実行例**:

```bash
# SySLLMで全体要約を行い、結果を自動評価
python main.py --method sysllm --evaluate

# MCTSでStep 150の行動理由を生成し、自動評価
python main.py --method mcts --step 150 --evaluate

# TalkToAgentで対話的分析を実行
python main.py --method talktoagent --query "Step 150でなぜスロットルを下げたのか検証して"

```
