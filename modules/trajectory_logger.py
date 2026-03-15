"""軌跡ロガー。

LAG の SingleCombatEnv をラップし、各ステップの状態・操舵入力を
XRL システムが読めるフォーマットの CSV に記録する。

使い方（LAG/envs/JSBSim/test/ 内のスクリプトから）::

    import sys, os
    sys.path.append(...)  # xrl_system を sys.path に追加済みであること
    from modules.trajectory_logger import TrajectoryLogger

    logger = TrajectoryLogger(env, ego_agent_index=0)
    while True:
        obs, reward, done, info = env.step(actions)
        logger.log_step()
        if done: break
    logger.save("data/trajectory_log.csv")

注意:
    LAG の envs.JSBSim が sys.path に含まれている必要があります。
    LAG/envs/JSBSim/test/ のスクリプトは自動的に LAG/ をパスに追加します。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from envs.JSBSim.core.catalog import Catalog as c
    from envs.JSBSim.utils.utils import get_AO_TA_R
    _LAG_AVAILABLE = True
except ImportError:
    _LAG_AVAILABLE = False

_MPS_TO_KT = 1.94384


class TrajectoryLogger:
    """SingleCombatEnv の step ごとに XRL 用カラムを記録するラッパー。

    記録するカラム:
        step, altitude, speed, distance, ata, aspect_angle,
        aileron, elevator, throttle
    """

    def __init__(self, env, ego_agent_index: int = 0) -> None:
        """
        Args:
            env:              SingleCombatEnv インスタンス
            ego_agent_index:  自機として扱うエージェントのインデックス (0 or 1)
        """
        if not _LAG_AVAILABLE:
            raise ImportError(
                "LAG の envs.JSBSim が見つかりません。"
                "LAG/ を sys.path に追加してください。"
            )
        self.env = env
        self.ego_idx = ego_agent_index
        self._rows: list[dict] = []

        # プロパティ参照 (インポート成功時のみ)
        self._props = {
            "altitude": c.position_h_sl_m,
            "speed_mps": c.velocities_vc_mps,
            "aileron":   c.fcs_aileron_cmd_norm,
            "elevator":  c.fcs_elevator_cmd_norm,
            "throttle":  c.fcs_throttle_cmd_norm,
        }

    def log_step(self) -> None:
        """現在のステップを記録する。env.step() の直後に呼ぶ。"""
        agents = list(self.env.agents.keys())
        if len(agents) < 2:
            return

        ego_uid = agents[self.ego_idx]
        enm_uid = agents[(self.ego_idx + 1) % 2]
        ego_sim = self.env.agents[ego_uid]
        enm_sim = self.env.agents[enm_uid]

        ego_feature = (*ego_sim.get_position(), *ego_sim.get_velocity())
        enm_feature = (*enm_sim.get_position(), *enm_sim.get_velocity())

        ata_rad, ta_rad, distance = get_AO_TA_R(ego_feature, enm_feature)

        self._rows.append({
            "step":         self.env.current_step,
            "altitude":     round(float(ego_sim.get_property_value(self._props["altitude"])), 2),
            "speed":        round(float(ego_sim.get_property_value(self._props["speed_mps"])) * _MPS_TO_KT, 2),
            "distance":     round(float(distance), 2),
            "ata":          round(float(np.degrees(ata_rad)), 3),
            "aspect_angle": round(float(np.degrees(ta_rad)), 3),
            "aileron":      round(float(ego_sim.get_property_value(self._props["aileron"])), 4),
            "elevator":     round(float(ego_sim.get_property_value(self._props["elevator"])), 4),
            "throttle":     round(float(ego_sim.get_property_value(self._props["throttle"])), 4),
        })

    def save(self, path: str = "data/trajectory_log.csv") -> pd.DataFrame:
        """記録済みデータを CSV に保存する。"""
        if not self._rows:
            print("[TrajectoryLogger] 記録がありません。log_step() を呼びましたか？")
            return pd.DataFrame()
        df = pd.DataFrame(self._rows)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"[TrajectoryLogger] {len(df)} ステップを保存: {out}")
        return df

    def reset(self) -> None:
        """記録をリセットする（新エピソード開始時）。"""
        self._rows.clear()

    @property
    def dataframe(self) -> pd.DataFrame:
        """現在の記録を DataFrame として返す。"""
        return pd.DataFrame(self._rows)
