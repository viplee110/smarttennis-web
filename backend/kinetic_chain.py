"""
kinetic_chain.py — 从 MediaPipe world landmarks 重建动力链分析
============================================================
输入: extract_landmarks.py 产出的 JSON ({fps, frames:[{img, world}]})
输出: 各环节归一化角速度时序、X-factor、contact 帧、诊断指标

坐标系 (MediaPipe world landmarks): 原点≈髋中点, x→右, y→下, z→朝摄像头。
垂直轴为 y, 水平面为 x-z。绕垂直轴的旋转 = 在 x-z 平面投影后的夹角。
"""
from __future__ import annotations
import numpy as np

# MediaPipe Pose 33 点中我们用到的索引
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24


def _smooth(x: np.ndarray, win: int = 5) -> np.ndarray:
    """简单滑动平均（边缘用 reflect 填充），抑制 MediaPipe 抖动。"""
    if win < 2 or len(x) < win:
        return x
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, pad, mode="reflect")
    k = np.ones(win) / win
    return np.convolve(xp, k, mode="valid")


def _planar_angle(p_left: np.ndarray, p_right: np.ndarray) -> np.ndarray:
    """两点连线在水平 (x-z) 平面内相对 x 轴的角度 (rad)，逐帧。"""
    d = p_right - p_left                      # (T,3)
    return np.unwrap(np.arctan2(d[:, 2], d[:, 0]))


def _seg_angular_speed(p_a: np.ndarray, p_b: np.ndarray, fps: float) -> np.ndarray:
    """肢段 a→b 的三维角速度大小 (rad/s): 单位方向向量的变化率。"""
    v = p_b - p_a
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1e-9
    u = v / n                                 # 单位方向 (T,3)
    du = np.gradient(u, axis=0) * fps         # d(u)/dt
    return np.linalg.norm(du, axis=1)


def load_world(data: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """从 JSON dict 取出 world 关节点数组 (T,33,3)、有效帧掩码、fps。
    缺检测的帧用前后线性插值补齐。"""
    fps = float(data.get("fps", 30.0))
    frames = data["frames"]
    T = len(frames)
    arr = np.full((T, 33, 3), np.nan)
    for i, f in enumerate(frames):
        if f.get("world"):
            arr[i] = np.array(f["world"], dtype=float)[:, :3]
    valid = ~np.isnan(arr[:, 0, 0])
    # 逐关节逐轴线性插值补 NaN
    idx = np.arange(T)
    for j in range(33):
        for k in range(3):
            col = arr[:, j, k]
            m = ~np.isnan(col)
            if m.sum() >= 2:
                arr[:, j, k] = np.interp(idx, idx[m], col[m])
            elif m.sum() == 1:
                arr[:, j, k] = col[m][0]
            else:
                arr[:, j, k] = 0.0
    return arr, valid, fps


def _wrist_speed(world: np.ndarray, j: int, fps: float) -> np.ndarray:
    return np.linalg.norm(np.gradient(world[:, j], axis=0) * fps, axis=1)


def _center_weight(T: int, sigma_frac: float = 0.18) -> np.ndarray:
    """以片段正中为峰的高斯权重 (smart_cutter 把挥拍放在片段中心),
    用来压低边缘的准备/恢复快动作, 突出真正的击球。"""
    idx = np.arange(T)
    c, sigma = (T - 1) / 2.0, sigma_frac * T
    return np.exp(-0.5 * ((idx - c) / sigma) ** 2)


def detect_handedness(world: np.ndarray, fps: float) -> str:
    """挥拍手 = 中心加权手腕速度峰值更高的那只手。"""
    w = _center_weight(world.shape[0])
    rp = (_wrist_speed(world, R_WR, fps) * w).max()
    lp = (_wrist_speed(world, L_WR, fps) * w).max()
    return "R" if rp >= lp else "L"


def compute_signals(world: np.ndarray, fps: float, hand: str = "auto") -> dict:
    """计算动力链各环节信号 (原始 + 峰值归一化) 与 X-factor。"""
    if hand == "auto":
        hand = detect_handedness(world, fps)
    sh = R_SH if hand == "R" else L_SH
    el = R_EL if hand == "R" else L_EL
    wr = R_WR if hand == "R" else L_WR

    T = world.shape[0]
    t = np.arange(T) / fps

    # 髋线 / 肩线 绕垂直轴的角度 (rad) → 角速度 (rad/s)
    hip_ang = _smooth(_planar_angle(world[:, L_HIP], world[:, R_HIP]))
    sho_ang = _smooth(_planar_angle(world[:, L_SH], world[:, R_SH]))
    hip_av = np.abs(np.gradient(hip_ang) * fps)
    sho_av = np.abs(np.gradient(sho_ang) * fps)

    # 上臂 / 前臂 三维角速度
    upper = _seg_angular_speed(world[:, sh], world[:, el], fps)
    fore = _seg_angular_speed(world[:, el], world[:, wr], fps)

    # 手腕线速度 (m/s)
    wr_v = np.linalg.norm(np.gradient(world[:, wr], axis=0) * fps, axis=1)

    # X-factor: 肩-髋 在水平面内的原始分离角 (deg)。中立站姿两线近似平行→≈0,
    # 装载期肩转多于髋→分离增大。不减任意基线, 避免片段首帧站姿差异引入噪声。
    xfactor = np.degrees(sho_ang - hip_ang)

    sig = {
        "hip": _smooth(hip_av), "shoulder": _smooth(sho_av),
        "upper_arm": _smooth(upper), "forearm": _smooth(fore),
        "wrist": _smooth(wr_v),
    }
    norm = {k: (v / v.max() if v.max() > 0 else v) for k, v in sig.items()}
    return {
        "t": t, "fps": fps, "hand": hand,
        "raw": sig, "norm": norm,
        "xfactor": _smooth(xfactor),
    }


def detect_contact(signals: dict) -> int:
    """击球瞬间 = 中心加权手腕线速度峰值帧 (避开恢复/准备的快动作)。"""
    w = signals["raw"]["wrist"]
    return int(np.argmax(w * _center_weight(len(w))))


def compute_metrics(signals: dict, contact: int) -> dict:
    """提炼诊断指标 (与 demo 报告下排三图一致)。"""
    tarr = signals["t"]
    fps = signals["fps"]

    # 近端→远端时序: 只在 contact 前后的发力窗口内找峰 (避开准备/随挥的杂峰),
    # 比较髋角速度峰 → 前臂角速度峰 的时间差 (s)。
    w0 = max(0, contact - int(round(0.6 * fps)))
    w1 = min(len(tarr), contact + int(round(0.15 * fps)) + 1)
    seg = slice(w0, w1)
    hip_pk = w0 + int(np.argmax(signals["raw"]["hip"][seg]))
    fore_pk = w0 + int(np.argmax(signals["raw"]["forearm"][seg]))
    lag_s = tarr[fore_pk] - tarr[hip_pk]
    # 归一化: 0.20s 视为理想满分窗口 (髋显著领先前臂 = 良好动力链)
    hip_to_forearm_lag = float(np.clip(lag_s / 0.20, 0.0, 1.0))

    xf = signals["xfactor"]
    pre = xf[:contact + 1] if contact > 0 else xf
    # X-factor 装载幅度 = 击球前的最大分离绝对值
    xfactor_magnitude = float(np.max(np.abs(pre)))
    # X-factor 释放 = 击球瞬间残留的分离 (deg)
    xfactor_release = float(xf[contact]) if contact < len(xf) else float(xf[-1])

    return {
        "hip_to_forearm_lag": hip_to_forearm_lag,
        "xfactor_magnitude": xfactor_magnitude,
        "xfactor_release": xfactor_release,
        "contact_frame": int(contact),
        "contact_t": float(tarr[contact]) if contact < len(tarr) else float(tarr[-1]),
        "hip_peak_t": float(tarr[hip_pk]),
        "forearm_peak_t": float(tarr[fore_pk]),
    }


def _crop_signals(signals: dict, lo: int, hi: int) -> dict:
    """把全段信号裁到挥拍窗口 [lo,hi)，并在窗口内重新做峰值归一化。"""
    raw = {k: v[lo:hi] for k, v in signals["raw"].items()}
    norm = {k: (v / v.max() if v.max() > 0 else v) for k, v in raw.items()}
    return {
        "t": signals["t"][lo:hi], "fps": signals["fps"], "hand": signals["hand"],
        "raw": raw, "norm": norm, "xfactor": signals["xfactor"][lo:hi],
        "window": [int(lo), int(hi)],
    }


def analyze(data: dict, hand: str = "auto",
            pre_s: float = 1.0, post_s: float = 0.7) -> dict:
    """端到端: landmarks JSON → 挥拍窗口内的信号 + contact + 指标。

    smart_cutter 切出的片段含准备与随挥; 我们以中部最快手腕峰为 contact,
    取其前 pre_s 秒、后 post_s 秒为挥拍窗口, 在窗口内做时序与指标分析。
    """
    world, valid, fps = load_world(data)
    full = compute_signals(world, fps, hand)
    contact = detect_contact(full)                      # 全段帧索引
    lo = max(0, contact - int(round(pre_s * fps)))
    hi = min(world.shape[0], contact + int(round(post_s * fps)) + 1)
    signals = _crop_signals(full, lo, hi)
    local_contact = contact - lo                        # 窗口内索引
    metrics = compute_metrics(signals, local_contact)
    metrics["contact_frame"] = int(contact)             # 覆盖为全段帧
    metrics["contact_t"] = float(contact / fps)
    return {"signals": signals, "contact": int(contact),
            "contact_local": int(local_contact), "world": world,
            "metrics": metrics, "valid_ratio": float(valid.mean())}
