"""
Two‑finger UMI gripper driven by a single tendon.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class UMIGripper(GripperModel):
    """自訂兩指夾爪（對稱滑動）"""

    def __init__(self, idn=0):
        # xml_path 以 robosuite/models/assets 為根目錄
        super().__init__(xml_path_completion("grippers/umi_gripper.xml"), idn=idn)

        # === 名稱必須和 XML 一致 ===
        self._joint_names = ("left_finger_joint", "right_finger_joint")
        self._actuator_names = ("fingers_actuator",)

        # 關節極限（來自 <joint range="0 0.05">）
        self._qlim_low, self._qlim_high = 0.0, 0.05

        # 初始姿態：完全張開
        self._init_qpos = np.array([self._qlim_high])

    # ------------------------------------------------------------------ #
    # 必要屬性
    # ------------------------------------------------------------------ #

    @property
    def dof(self):
        # 只有 1 個自由度（由 tendon 同步兩個滑動關節）
        return 1

    @property
    def bottom_offset(self):
        """
        夾爪基座到抓取平面的位移（m，夾爪座標系）。
        XML 中 `eef` body 位於 (0, ‑0.18, 0)，直接取這個值。
        """
        return np.array([0.0, -0.18, 0.0])

    @property
    def init_qpos(self):
        return self._init_qpos

    # ------------------------------------------------------------------ #
    # 動作格式化：[-1, 1] → 關節／tendon 目標
    # ------------------------------------------------------------------ #

    def format_action(self, action):
        """
        外部指令 action ∈ [-1, 1]：
            +1 → 完全閉合（tendon = 0.0）
            -1 → 完全張開（tendon = 0.05）
        返回值為 MuJoCo ctrl 陣列（np.ndarray, shape=(1,)）
        """
        action = np.clip(action, -1.0, 1.0)

        # [-1,1] 線性映射到 [open, close] → [0.05, 0.0]
        pos_frac = 0.5 * (1.0 - action[0])          # [-1,1] → [1,0]
        tendon_len = self._qlim_low + pos_frac * (self._qlim_high - self._qlim_low)

        return np.array([tendon_len])
