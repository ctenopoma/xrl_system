# 評価の手順

学習なし (外部 API baseline) と学習あり (LoRA) の両パターンで
Soundness / Fidelity を測定し、結果を比較する方法をまとめる。

---

## 前提

```
XRL_API_KEY=<your-api-key>          # Judge (評価) 用の API キー
XRL_MODEL_NAME=gpt-4o               # 省略時のデフォルト
```

データは `data/trajectory_log.csv` に存在すること。

---

## 1. 学習なし評価 (baseline / 外部 API)

LoRA アダプタなしで外部 LLM を推論バックエンドとして使う。
これがベースライン (比較基準) になる。

```bash
# ステップ 50, 100, 150 を v1_basic テンプレート・zero_shot で評価
python evaluate_baseline.py --steps 50 100 150

# テンプレートと戦略を指定
python evaluate_baseline.py --steps 50 100 150 \
    --template v1_combat_only --strategy cot

# ランダムに 20 ステップをサンプリングして評価
python evaluate_baseline.py --sample 20

# 複数テンプレートを一括比較
python evaluate_baseline.py --steps 100 150 \
    --template v1_basic v1_combat_only v2_with_prior

# 事前情報あり (SySLLM 要約を注入)
python evaluate_baseline.py --steps 50 100 150 \
    --template v2_with_prior --prior-info sysllm
```

結果は `results/baseline/baseline_summary.csv` に **追記** される。

---

## 2. 学習あり評価 (LoRA アダプタ使用)

### 2-1. データセット生成

```bash
# CoT ラベル、スコアフィルタなし
python build_dataset.py --label-source cot

# MCTS ラベル、スコア 3.0 以上のみ収集
python build_dataset.py --label-source mcts --min-score 3.0 \
    --template v1_combat_only --mcts-iterations 6

# 出力先: datasets/<dataset_id>/
#   train.jsonl, metadata.json
```

### 2-2. LoRA 学習

```bash
# 基本実行 (rank=16, epoch=3, lr=2e-4)
python train_lora.py --dataset datasets/<dataset_id>

# rank・epoch を変更
python train_lora.py --dataset datasets/<dataset_id> \
    --rank 8 --epochs 5

# GPU メモリを節約したい場合 (4-bit 量子化)
python train_lora.py --dataset datasets/<dataset_id> \
    --load-in-4bit

# 出力先: models/<run_id>/adapter/
#   adapter_config.json, adapter_model.safetensors など
```

`<run_id>` は学習完了時のログに `run_id :` として出力される。

### 2-3. 学習後評価

`--backend local` と `--adapter` を指定する。
それ以外のオプションは学習なし評価と同じ。

```bash
# ステップ 50, 100, 150 をアダプタで評価
python evaluate_baseline.py --steps 50 100 150 \
    --backend local \
    --adapter models/<run_id>/adapter

# ベースモデルを明示指定 (run_config.json がない場合)
python evaluate_baseline.py --steps 50 100 150 \
    --backend local \
    --adapter models/<run_id>/adapter \
    --base-model Qwen/Qwen2.5-7B-Instruct

# テンプレートは学習時と合わせる
python evaluate_baseline.py --steps 50 100 150 \
    --backend local \
    --adapter models/<run_id>/adapter \
    --template v1_combat_only --strategy cot
```

結果は `results/baseline/baseline_summary.csv` に追記される
(`backend` 列が `local` になる)。

---

## 3. 結果の比較

```bash
# 比較レポートを Markdown で生成
python compare_results.py \
    --summary results/baseline/baseline_summary.csv \
    --output  results/comparison_report.md

# stdout に表示するだけ
python compare_results.py \
    --summary results/baseline/baseline_summary.csv

# baseline 行だけ確認したい場合
python compare_results.py \
    --summary results/baseline/baseline_summary.csv \
    --show-baseline-only
```

出力される `comparison_report.md` の構成:

| 列 | 内容 |
|----|------|
| `backend` | `(baseline)` = 外部 API / `local` = LoRA |
| `template_id` | 使用したプロンプトテンプレート |
| `strategy` | `zero_shot` / `cot` |
| `prior_info_mode` | `none` / `sysllm` |
| `soundness` | 行動の妥当性スコア (平均) |
| `fidelity` | 方針との一致度スコア (平均) |
| `total_mean` | (soundness + fidelity) / 2 |
| `vs baseline` | LoRA 適用後の total_mean の差分 |

---

## 4. 複数設定を一括実行 (スイープ)

`configs/sweep_example.yaml` を編集してからスイープを実行すると、
データセット生成・学習・評価が全組み合わせで自動実行される。

```bash
# dry-run で combo 一覧を確認
python run_experiment.py --config configs/sweep_example.yaml --dry-run

# 実行 (baseline 評価も含める場合は --eval-baseline を追加)
python run_experiment.py --config configs/sweep_example.yaml --eval-baseline

# 途中で中断した場合は --resume で再開
python run_experiment.py --config configs/sweep_example.yaml \
    --resume experiments/sweep_<timestamp>
```

結果はすべて `results/baseline/baseline_summary.csv` に追記され、
`compare_results.py` で比較できる。

---

## 5. 典型的なワークフロー

```
# ① baseline を測定
python evaluate_baseline.py --steps 50 100 150 180

# ② データセット生成
python build_dataset.py --label-source mcts --min-score 3.0

# ③ 学習
python train_lora.py --dataset datasets/<dataset_id> --rank 16

# ④ 学習後評価 (同じステップ・テンプレートを使う)
python evaluate_baseline.py --steps 50 100 150 180 \
    --backend local --adapter models/<run_id>/adapter

# ⑤ 比較
python compare_results.py \
    --summary results/baseline/baseline_summary.csv \
    --output  results/comparison_report.md
```
