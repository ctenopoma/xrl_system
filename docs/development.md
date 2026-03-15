# XRL System 改修ガイド

このドキュメントは、XRL System を改修・拡張したい開発者向けのガイドです。
各モジュールの役割、改修ポイント、よくある変更パターンを説明します。

---

## 目次

1. [アーキテクチャ概要](#1-アーキテクチャ概要)
2. [モジュール別改修ガイド](#2-モジュール別改修ガイド)
3. [よくある改修パターン](#3-よくある改修パターン)
4. [プロンプトのチューニング](#4-プロンプトのチューニング)
5. [新しい分析モードの追加手順](#5-新しい分析モードの追加手順)
6. [テスト方法](#6-テスト方法)

---

## 1. アーキテクチャ概要

```
main.py                     ← CLI エントリーポイント (引数解析・出力整形)
  │
  ├── modules/llm_client.py ← 全モジュールが依存する共通 LLM クライアント
  ├── modules/data_loader.py← CSV 読み込み・ダミー生成・前処理
  │
  ├── modules/sysllm.py     ← モード2: 全体要約 (SySLLM)
  ├── modules/talktoagent.py← モード3: マルチエージェント対話
  ├── modules/mcts_xrl.py   ← モード1 & 4: CoT / MCTS-XRL 局所説明
  └── modules/evaluator.py  ← LLM-as-a-Judge 自動評価
```

### 依存関係の方向

```
llm_client ◄── sysllm
           ◄── talktoagent
           ◄── mcts_xrl
           ◄── evaluator

data_loader◄── sysllm
           ◄── talktoagent
           ◄── mcts_xrl
           ◄── main
```

> **重要**: `llm_client.py` と `data_loader.py` は基盤モジュールです。
> 他のモジュールはこの2つにのみ依存します。循環依存を避けてください。

---

## 2. モジュール別改修ガイド

### `modules/llm_client.py` — LLM クライアント

**改修ポイント:**

- **別のモデルを使いたい**: 環境変数 `XRL_MODEL_NAME` か `--model` CLI 引数で変更するだけでよく、コード変更は不要です。
- **タイムアウトや retry を設定したい**: `chat()` メソッド内の `litellm.completion()` 呼び出しに `timeout` や `num_retries` を追加してください。
  ```python
  response = litellm.completion(**kwargs, timeout=60, num_retries=3)
  ```
- **ローカルモデル (Ollama など) を使いたい**: `XRL_MODEL_NAME="ollama/llama3"` のように設定するだけで litellm が対応します。
- **Judge 用に別モデルを指定したい**: `EpisodeEvaluator` の初期化時に別の `LLMClient` インスタンスを渡してください。
  ```python
  judge_llm = LLMClient(model="gpt-4-turbo")
  evaluator = EpisodeEvaluator(judge_llm)
  ```

---

### `modules/data_loader.py` — データローダー

**改修ポイント:**

- **CSV のカラムを追加・変更したい**:
  - `COLUMNS`, `STATE_COLS`, `ACTION_COLS` 定数を更新する。
  - `to_trajectory_text()` のフォーマット文字列を更新する。
  - `_generate_dummy_data()` に対応するダミー値生成を追加する。

- **キーフレームの抽出ロジックを変更したい**: `filter_keyframes()` 内の条件を変更する。
  - `ATA_ATTACK_THRESHOLD`, `MANEUVER_THRESHOLD`, `DISTANCE_CHANGE_PERCENTILE` の定数を変えるだけで閾値調整可能。
  - 新しい抽出条件（例：speed が急変したフレーム）を `combined_mask` に追加する。

- **エピソードが複数ある（マルチエピソード対応）**: `csv_path` を受け取る部分を拡張し、複数ファイルを結合した DataFrame を返すように `load()` を改修する。

---

### `modules/sysllm.py` — SySLLM

**改修ポイント:**

- **出力セクションを変更したい**:
  - `SUMMARY_KEYS` リストを変更する。
  - `_SYSTEM_PROMPT` 内の「出力フォーマット」部分を対応して変更する。
  - `_parse_response()` のヘッダーマップ (`header_map`) を変更する。

- **JSON 形式で出力させたい**: `_build_prompt()` にJSON出力指示を追加し、`analyze()` 内で `response_format={"type": "json_object"}` を渡すように変更する。

---

### `modules/talktoagent.py` — TalkToAgent

**改修ポイント:**

- **sandbox に追加ライブラリを許可したい**: `_SAFE_BUILTINS` 辞書に追加するか、`_execute_code()` の `safe_globals` に追加する。
  ```python
  import numpy as np
  safe_globals["np"] = np  # numpy を許可
  ```
  > ⚠️ セキュリティリスクに注意。`os`, `subprocess`, `sys` などは絶対に許可しないこと。

- **リトライ回数を変更したい**: `MAX_DEBUG_RETRIES` 定数を変更する（デフォルト: 3）。

- **各エージェントのプロンプトを変更したい**: `_coordinator()`, `_coder()`, `_explainer()` 内のシステムプロンプト文字列を直接変更する。

- **エージェント間でメモリ（会話履歴）を共有したい**: 各エージェントが `messages` リストに追記していく形に変更し、`self.llm.chat(messages)` で渡す。

---

### `modules/mcts_xrl.py` — MCTS-XRL

**改修ポイント:**

- **MCTS の探索戦略を変えたい**:
  - UCB1 定数を変更: `UCB_CONSTANT = math.sqrt(2)` の値を調整する。
  - 探索ポリシーを変えたい場合: `_select()` メソッドを改修する（例：epsilon-greedy 探索）。

- **ノードの展開数（branching factor）を増やしたい**: `explain_mcts()` のループ内で、1イテレーションあたり複数の子ノードを生成するように変更する。

- **MCTSNode に追加情報を持たせたい**: `@dataclass MCTSNode` にフィールドを追加する（例：`generation_time: float`）。

- **自己評価スコアの基準を変えたい**: `_evaluator_score()` のシステムプロンプトを変更する。指標を3軸にしたい場合は戻り値の最大値も合わせて変更する。

---

### `modules/evaluator.py` — 評価モジュール

**改修ポイント:**

- **評価指標を追加・変更したい**: `_JUDGE_SYSTEM` 定数内の評価指標説明を変更し、`evaluate()` 内の値域チェックも更新する。

- **評価結果をファイルに保存したい**: `evaluate()` の戻り値を受け取った後に保存処理を追加するか、`evaluate_batch()` 内でまとめて CSV 保存するロジックを追加する。

- **JSON パース失敗時のデフォルト値を変えたい**: `evaluate()` の例外処理ブロックでデフォルト値を返すように変更する。
  ```python
  except ValueError:
      return {"soundness": 0, "fidelity": 0, "reason": "評価失敗"}
  ```

---

### `main.py` — CLI エントリーポイント

**改修ポイント:**

- **新しい CLI 引数を追加したい**: `build_parser()` に `p.add_argument(...)` を追加し、`run()` 内で参照する。

- **出力をファイルに保存したい**: `run()` の最後に `result` 辞書を JSON/CSV で保存する処理を追加する。`--output` 引数を追加するのが自然な拡張です。

- **複数手法を一括実行したい**: `run()` を呼ぶループを追加し、全モードの結果を集約して比較テーブルを出力する `compare` モードを追加する。

---

## 3. よくある改修パターン

### パターン1: モデルを変える

コード変更なし。環境変数で対応:
```bash
XRL_MODEL_NAME="claude-3-5-sonnet-20241022" python main.py --method sysllm
```

### パターン2: CSVの列を追加する

1. `data_loader.py` の `COLUMNS`, `STATE_COLS` or `ACTION_COLS` に追加
2. `to_trajectory_text()` の f-string に追加
3. `_generate_dummy_data()` にダミー値の生成を追加
4. 各モジュールのプロンプト（`_STATE_DESCRIPTION` 等）に説明を追加

### パターン3: 評価指標を追加する

1. `evaluator.py` の `_JUDGE_SYSTEM` に新指標を追記
2. `evaluate()` の値域チェックに新指標を追加
3. `mcts_xrl.py` の `_evaluator_score()` も同様に更新（Q値スケールが変わるため）
4. `main.py` の `_print_eval()` に新指標の表示を追加

### パターン4: 新モードを追加する

次セクション「[新しい分析モードの追加手順](#5-新しい分析モードの追加手順)」を参照してください。

---

## 4. プロンプトのチューニング

各モジュールのプロンプトは定数または関数内の文字列として管理されています。改修時のポイント:

| モジュール       | プロンプト定義場所                    |
|---------------|-------------------------------------|
| sysllm        | `_SYSTEM_PROMPT` 定数 + `_build_prompt()` |
| talktoagent   | 各 `_coordinator()`, `_coder()`, `_explainer()` メソッド内 |
| mcts_xrl      | `_STATE_DESCRIPTION` 定数 + 各ロールメソッド内 |
| evaluator     | `_JUDGE_SYSTEM` 定数 + `_build_judge_prompt()` |

**チューニングのコツ:**

- データの意味（単位、範囲、正負の意味）を必ず system prompt に含める。
- 出力フォーマット（JSON か自然言語か、箇条書きかセクション分けか）を明示する。
- Few-shot 例を追加すると品質が向上する。ただしトークン消費が増加するためトレードオフに注意。

---

## 5. 新しい分析モードの追加手順

例として「モード5: RAG-XRL（検索拡張生成）」を追加する場合:

### ステップ1: モジュールを作成する

```python
# modules/rag_xrl.py
from modules.llm_client import LLMClient
from modules.data_loader import DataLoader

class RAGXRL:
    def __init__(self, llm: LLMClient, loader: DataLoader) -> None:
        ...

    def explain(self, step: int) -> dict:
        """説明を返す。"explanation" キーを必ず含めること（evaluator との互換性）"""
        ...
        return {"step": step, "explanation": "...", ...}
```

> `result["explanation"]` キーを持つ辞書を返すことが `--evaluate` との互換性に必要です。

### ステップ2: `main.py` に追加する

```python
# インポートを追加
from modules.rag_xrl import RAGXRL

# build_parser() の choices に追加
p.add_argument("--method", choices=["cot", "sysllm", "talktoagent", "mcts", "rag"])

# run() に分岐を追加
elif args.method == "rag":
    _require_step(args)
    result = RAGXRL(llm, loader).explain(args.step)
    _print_local_result("RAG-XRL", result)
```

### ステップ3: ドキュメントを更新する

`docs/usage.md` に新モードの使い方を追記してください。

---

## 6. テスト方法

### ダミーデータで動作確認する

CSV なしで実行すると自動的にダミーデータ（200ステップ）が使われます:

```bash
python main.py --method cot --step 100
python main.py --method sysllm
```

### LLM API なしで単体テストする

`LLMClient` をモックに差し替えることで API コストなしにテスト可能です:

```python
from unittest.mock import MagicMock
from modules.llm_client import LLMClient
from modules.sysllm import SySLLM
from modules.data_loader import DataLoader

mock_llm = MagicMock(spec=LLMClient)
mock_llm.simple_prompt.return_value = """
## 1. 戦術的アプローチ (tactical_approach)
テスト用の応答です。
## 2. 状況への適応 (situational_adaptation)
テスト用の応答です。
## 3. 非効率性と弱点 (inefficiencies)
テスト用の応答です。
## 4. 総合要約 (overall_summary)
テスト用の応答です。
"""

loader = DataLoader("data/trajectory_log.csv")  # ダミー生成される
result = SySLLM(mock_llm, loader).analyze()
assert result["tactical_approach"] != ""
```

### `data_loader.py` のテスト

DataLoader はダミーデータを生成できるため API 不要で完全にテスト可能:

```python
from modules.data_loader import DataLoader

loader = DataLoader("nonexistent.csv")  # ダミー生成される
df = loader.load()
assert len(df) == 200
assert list(df.columns) == ["step", "altitude", ...]

ctx = loader.get_step_context(100)
assert ctx["step"] == 100
assert "altitude" in ctx["state"]

keyframes = loader.filter_keyframes()
assert len(keyframes) <= len(df)
```
