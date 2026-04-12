"""データローダー。

trajectory_log.csv の読み込み・ダミー生成・フィルタリング・テキスト化を担う。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# カラム定義
COLUMNS = ["step", "altitude", "speed", "distance", "ata", "aspect_angle",
           "aileron", "elevator", "throttle"]
STATE_COLS = ["altitude", "speed", "distance", "ata", "aspect_angle"]
ACTION_COLS = ["aileron", "elevator", "throttle"]

# StepContext 型エイリアス
StepContext = dict  # {"step": int, "state": dict, "action": dict}

# キーフレーム抽出の閾値
ATA_ATTACK_THRESHOLD = 45.0        # 攻撃機会と判定するata度数
MANEUVER_THRESHOLD = 0.5           # 急機動と判定する操舵変化量
DISTANCE_CHANGE_PERCENTILE = 90    # 距離急変と判定するパーセンタイル


class DataLoader:
    """CSV 読み込み・ダミー生成・前処理を担う。

    使い方::

        loader = DataLoader("data/trajectory_log.csv")
        df = loader.load()                   # CSV 読み込み (なければダミー生成)
        text = loader.to_trajectory_text()   # テキスト化
        ctx = loader.get_step_context(150)   # 特定ステップの状態取得
    """

    def __init__(self, csv_path: str = "data/trajectory_log.csv") -> None:
        self.csv_path = Path(csv_path)
        self._df: Optional[pd.DataFrame] = None  # 遅延ロード

    # ------------------------------------------------------------------
    # 公開API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """CSV を読み込む。ファイルが存在しない場合はダミーデータを生成して返す。

        Returns:
            COLUMNS を持つ DataFrame (step 列でソート済み)
        """
        if self._df is not None:
            return self._df

        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            # 不足カラムがあれば 0 埋め
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[COLUMNS].sort_values("step").reset_index(drop=True)
        else:
            print(f"[DataLoader] {self.csv_path} が見つかりません。ダミーデータを生成します。")
            df = self._generate_dummy_data()

        self._df = df
        return self._df

    def get_step_context(self, step: int) -> StepContext:
        """指定ステップの状態・行動を辞書で返す。

        Args:
            step: 取得するステップ番号

        Returns:
            {"step": int,
             "state": {"altitude": float, ...},
             "action": {"aileron": float, ...}}

        Raises:
            ValueError: step が存在しない場合
        """
        df = self.load()
        rows = df[df["step"] == step]
        if rows.empty:
            available = sorted(df["step"].unique().tolist())
            raise ValueError(
                f"Step {step} は存在しません。利用可能な範囲: {available[0]}〜{available[-1]}"
            )
        row = rows.iloc[0]
        return {
            "step": int(row["step"]),
            "state": {col: float(row[col]) for col in STATE_COLS},
            "action": {col: float(row[col]) for col in ACTION_COLS},
        }

    def to_trajectory_text(
        self,
        df: Optional[pd.DataFrame] = None,
        max_rows: int = 50,
    ) -> str:
        """DataFrame を自然言語テキストに変換する。

        LLM プロンプトへの埋め込み用。max_rows を超える場合は等間隔サンプリング。

        Args:
            df:       対象 DataFrame (省略時は load() の結果全体)
            max_rows: 渡す最大行数

        Returns:
            "Step 1: 高度=3000m, 速度=250kt, ..." 形式の箇条書き文字列
        """
        if df is None:
            df = self.load()

        if len(df) > max_rows:
            indices = np.linspace(0, len(df) - 1, max_rows, dtype=int)
            df = df.iloc[indices]

        lines: list[str] = []
        for _, row in df.iterrows():
            line = (
                f"Step {int(row['step'])}: "
                f"高度={row['altitude']:.0f}m, "
                f"速度={row['speed']:.1f}kt, "
                f"距離={row['distance']:.0f}m, "
                f"ATA={row['ata']:.1f}°, "
                f"AspectAngle={row['aspect_angle']:.1f}°, "
                f"エルロン={row['aileron']:.2f}, "
                f"エレベータ={row['elevator']:.2f}, "
                f"スロットル={row['throttle']:.2f}"
            )
            lines.append(line)
        return "\n".join(lines)

    def filter_keyframes(
        self, df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """キーフレーム抽出 (LAG など他手法向け。SySLLM では使用しない)。

        以下のいずれかを満たす行を抽出する:
            - ata < ATA_ATTACK_THRESHOLD (攻撃機会)
            - |aileron|, |elevator|, |throttle| いずれかの差分 >= MANEUVER_THRESHOLD (急機動)
            - |distance| の差分が全体の上位 DISTANCE_CHANGE_PERCENTILE% (距離急変)

        Args:
            df: 対象 DataFrame (省略時は load() 結果)

        Returns:
            条件を満たす行のみの DataFrame (元の index を保持)
        """
        if df is None:
            df = self.load()

        # 攻撃機会
        mask_attack = df["ata"] < ATA_ATTACK_THRESHOLD

        # 急機動: 操舵の前行との差分
        action_diff = df[ACTION_COLS].diff().abs()
        mask_maneuver = (action_diff >= MANEUVER_THRESHOLD).any(axis=1)

        # 距離急変: distance の差分が上位N%
        dist_diff = df["distance"].diff().abs()
        threshold = np.percentile(dist_diff.dropna(), DISTANCE_CHANGE_PERCENTILE)
        mask_dist = dist_diff >= threshold

        combined_mask = mask_attack | mask_maneuver | mask_dist
        return df[combined_mask].copy()

    # ------------------------------------------------------------------
    # 非公開ヘルパー
    # ------------------------------------------------------------------

    def _generate_dummy_data(self, n_steps: int = 200) -> pd.DataFrame:
        """現実的なフライトシミュレーション風のダミーデータを生成する。

        Args:
            n_steps: 生成するステップ数

        Returns:
            COLUMNS を持つ DataFrame
        """
        rng = np.random.default_rng(42)
        steps = np.arange(1, n_steps + 1)

        # 高度: 3000m から徐々に変動
        altitude = 3000 + np.cumsum(rng.normal(0, 20, n_steps))
        altitude = np.clip(altitude, 500, 8000)

        # 速度: 250kt 前後
        speed = 250 + np.cumsum(rng.normal(0, 2, n_steps))
        speed = np.clip(speed, 150, 450)

        # 距離: 敵との距離 (徐々に接近・離脱を繰り返す)
        distance = 5000 + np.cumsum(rng.normal(-10, 100, n_steps))
        distance = np.clip(distance, 500, 15000)

        # ATA: 0〜180° (攻撃チャンスが時々発生)
        ata = np.abs(rng.normal(60, 40, n_steps))
        ata = np.clip(ata, 0, 180)

        # Aspect Angle: 0〜180°
        aspect_angle = np.abs(rng.normal(90, 30, n_steps))
        aspect_angle = np.clip(aspect_angle, 0, 180)

        # 操舵入力: -1〜1 の範囲、急機動を時々挿入
        aileron = np.clip(rng.normal(0, 0.1, n_steps), -1, 1)
        elevator = np.clip(rng.normal(0, 0.1, n_steps), -1, 1)
        throttle = np.clip(0.7 + rng.normal(0, 0.1, n_steps), 0, 1)

        # 急機動を数箇所挿入
        for idx in rng.integers(10, n_steps - 10, size=10):
            aileron[idx] = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0)
            elevator[idx] = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0)
            throttle[idx] = rng.uniform(0.1, 0.3)

        return pd.DataFrame({
            "step":         steps,
            "altitude":     altitude,
            "speed":        speed,
            "distance":     distance,
            "ata":          ata,
            "aspect_angle": aspect_angle,
            "aileron":      aileron,
            "elevator":     elevator,
            "throttle":     throttle,
        })
