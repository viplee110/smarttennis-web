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


def detect_facing(frames) -> float:
    """面朝方向: 鼻子相对双耳中点的水平偏移 (鼻在前)。
    +1 = 面朝图像右侧(+x), -1 = 面朝左侧。用于跨机位镜像对齐。"""
    NOSE, EAR_L, EAR_R = 0, 7, 8
    vals = []
    for f in frames:
        im = f.get("img")
        if not im:
            continue
        vals.append(im[NOSE][0] - (im[EAR_L][0] + im[EAR_R][0]) / 2.0)
    if not vals:
        return 1.0
    return 1.0 if float(np.mean(vals)) >= 0 else -1.0


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

    # X-factor: 肩-髋 在水平面内的分离角 (deg)。
    # 重要: 用"原始角度差再 wrap 到 ±180°", 而非两条 unwrap 角度相减——
    # 后者会因各自缠绕累积出几百度的垃圾值 (深度噪声所致)。
    hip_raw = np.arctan2((world[:, R_HIP] - world[:, L_HIP])[:, 2],
                         (world[:, R_HIP] - world[:, L_HIP])[:, 0])
    sho_raw = np.arctan2((world[:, R_SH] - world[:, L_SH])[:, 2],
                         (world[:, R_SH] - world[:, L_SH])[:, 0])
    dsep = sho_raw - hip_raw
    xfactor = np.degrees(np.arctan2(np.sin(dsep), np.cos(dsep)))   # wrap 到 [-180,180]
    xfactor *= (1.0 if hand == "R" else -1.0)   # 左手镜像规整, 与右手同约定可比

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


def _img_track(frames, j: int) -> np.ndarray:
    """取关节 j 的图像坐标 (x,y) 时序 (T,2)，缺帧线性插值。"""
    T = len(frames)
    a = np.full((T, 2), np.nan)
    for i, f in enumerate(frames):
        im = f.get("img")
        if im:
            a[i] = [im[j][0], im[j][1]]
    g = np.arange(T)
    for k in range(2):
        col = a[:, k]
        m = ~np.isnan(col)
        if m.sum() >= 2:
            a[:, k] = np.interp(g, g[m], col[m])
        elif m.sum() == 1:
            a[:, k] = col[m][0]
        else:
            a[:, k] = 0.0
    return a


def detect_swing(frames, fps: float, hand: str = "R", facing: float = None):
    """返回 (contact, forward_swing_start) 全段帧索引。

    forward_swing_start(引拍末端) = 手腕在挥击方向最靠后处;
    contact(击球, 近似) = 前挥窗口[起点,随挥最靠前]内手腕"向前速度"最大处。
    用图像坐标(可靠) + facing 投影出"向前"方向, 避免引拍反向快动作误判。
    纯姿态有 ~±0.4s 固有误差, 仅作自动起点, 由前端滑杆做绝对校正。
    """
    if facing is None:
        facing = detect_facing(frames)
    wr = R_WR if hand != "L" else L_WR
    wx = _smooth(_img_track(frames, wr)[:, 0], 5)
    T = len(frames)
    fpos = facing * wx                       # 向挥击方向的位置
    vf = facing * np.gradient(wx)            # 向挥击方向的速度
    back0 = int(np.argmin(fpos[: max(2, int(T * 0.6))]))  # 粗定前挥窗口
    fwd_end = int(np.argmax(fpos))                         # 随挥(最靠前)
    if fwd_end <= back0:
        fwd_end = T - 1
    contact = back0 + int(np.argmax(vf[slice(back0, fwd_end + 1)]))
    # 引拍起点(精): 仅在击球前 2s 内找手腕最靠后处, 避免早期准备动作被误当引拍
    lo_w = max(0, contact - int(round(2.0 * fps)))
    swing_start = lo_w + int(np.argmin(fpos[lo_w: contact + 1])) if contact > lo_w else lo_w
    return contact, swing_start


def detect_contact(frames, fps: float, hand: str = "R", facing: float = None) -> int:
    return detect_swing(frames, fps, hand, facing)[0]


def dtw_path(a: np.ndarray, b: np.ndarray) -> list:
    """两条多维时间序列 a(N,D)、b(M,D) 的 DTW 非线性对齐路径 [(i,j)...]。
    用于把用户与德约的挥拍逐帧对应(引拍↔引拍/击球↔击球/随挥↔随挥), 跨节奏。
    仅用于可视化/对应, 不用于打分(DTW 会 warp 掉要测的时序差异)。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    N, M = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)   # (N,M)
    acc = np.full((N + 1, M + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    i, j, path = N, M, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        c = min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
        if acc[i - 1, j - 1] == c:
            i, j = i - 1, j - 1
        elif acc[i - 1, j] == c:
            i -= 1
        else:
            j -= 1
    return path[::-1]


def dtw_ref_to_user(user_norm: dict, ref_curve: dict) -> dict:
    """返回 {德约窗口内索引 j: 对应的用户窗口内索引 i}。
    特征 = 5 环节归一化速度向量(逐帧)。德约用 ideal_curve 的(可能下采样)序列。"""
    keys = ["hip", "shoulder", "upper_arm", "forearm", "wrist"]
    ua = np.stack([np.asarray(user_norm[k], float) for k in keys], axis=1)
    rb = np.stack([np.asarray(ref_curve[k], float) for k in keys], axis=1)
    jmap = {}
    for i, j in dtw_path(ua, rb):
        jmap.setdefault(j, i)            # 每个德约帧取首个对应用户帧
    return jmap


def contact_point(world_contact, hand: str = "R") -> tuple:
    """击球点(身体坐标系, 躯干为单位, 视角无关): 持拍手腕相对髋中心的
    (前伸量, 高度)。用 3D world 推解剖坐标系 → 不受拍摄角度影响。
    前伸>0=在身体前方; 高度>0=高于髋。"""
    p = np.asarray(world_contact, dtype=float)[:, :3]
    hip_c = (p[L_HIP] + p[R_HIP]) / 2.0
    sh_c = (p[L_SH] + p[R_SH]) / 2.0
    p = p - hip_c
    right = p[R_HIP] - p[L_HIP]
    right = right / (np.linalg.norm(right) + 1e-9)
    up = sh_c - hip_c
    up = up - np.dot(up, right) * right
    up = up / (np.linalg.norm(up) + 1e-9)
    fwd = np.cross(right, up)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    scale = np.linalg.norm(sh_c - hip_c) or 1e-6
    wr = p[R_WR] if hand != "L" else p[L_WR]
    # 击球瞬间持拍手在身前, 取 |前伸| 使"越大=越靠前", 直观且与机位无关
    return round(abs(float(wr @ fwd)) / scale, 3), round(float(wr @ up) / scale, 3)


def compute_metrics(signals: dict, contact: int) -> dict:
    """提炼诊断指标 (与 demo 报告下排三图一致)。"""
    tarr = signals["t"]
    fps = signals["fps"]

    # 近端→远端时序: 只在 contact 前后的发力窗口内找峰 (避开准备/随挥的杂峰),
    # 比较髋角速度峰 → 前臂角速度峰 的时间差 (s)。
    w0 = max(0, contact - int(round(0.6 * fps)))
    w1 = min(len(tarr), contact + int(round(0.15 * fps)) + 1)
    seg = slice(w0, w1)
    # 五个环节各自的"发力时刻"(相对击球, 秒): 用强度加权质心而非 argmax,
    # 在低帧率+姿态抖动下远比单帧峰值稳定 (argmax 会让各环节挤成同一帧)。
    order = ["hip", "shoulder", "upper_arm", "forearm", "wrist"]
    tt = np.asarray(tarr[seg]) - tarr[contact]
    peak_times = {}
    for k in order:
        s = np.asarray(signals["raw"][k][seg], dtype=float)
        s = s / (s.max() + 1e-9)
        wts = np.clip(s - 0.5, 0.0, None) ** 2          # 只让"明显发力"的区间计入质心
        if wts.sum() < 1e-6:
            wts = s
        peak_times[k] = round(float((tt * wts).sum() / (wts.sum() + 1e-9)), 3)
    # 粗粒度近端→远端领先 (秒): 距端组(上臂/前臂/手腕)质心 − 近端组(髋/肩)质心。
    # 远端三环节峰差 <1帧、低于25-30fps分辨率不可靠, 故只取"组级"这个可分辨的信号。
    prox = (peak_times["hip"] + peak_times["shoulder"]) / 2.0
    dist = (peak_times["upper_arm"] + peak_times["forearm"] + peak_times["wrist"]) / 3.0
    prox_lead_s = round(float(dist - prox), 3)          # 正 = 近端先发力(好)

    xf = signals["xfactor"]
    pre = xf[:contact + 1] if contact > 0 else xf
    xfactor_magnitude = float(np.max(np.abs(pre)))     # 击球前最大肩髋分离

    return {
        "prox_lead_s": prox_lead_s,
        "xfactor_magnitude": xfactor_magnitude,
        "contact_frame": int(contact),
        "contact_t": float(tarr[contact]) if contact < len(tarr) else float(tarr[-1]),
        "peak_times": peak_times,
        "sequence_ok": bool(prox_lead_s > 0),
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


def analyze(data: dict, hand: str = "auto", contact_override: int = None,
            pre_s: float = 1.0, post_s: float = 0.7) -> dict:
    """端到端: landmarks JSON → 挥拍窗口内的信号 + contact + 指标。

    contact 默认用前挥"向前速度"峰自动检测; contact_override 不为 None 时
    (前端滑杆校正后) 直接采用指定帧, 其余下游(裁窗/指标/对齐)随之重算。
    """
    world, valid, fps = load_world(data)
    full = compute_signals(world, fps, hand)
    facing = detect_facing(data["frames"])
    auto_contact, swing_start = detect_swing(data["frames"], fps, full["hand"], facing)
    if contact_override is not None:
        contact = int(max(0, min(world.shape[0] - 1, contact_override)))
    else:
        contact = auto_contact
    # 装载时长 = 前挥起点→击球, 用于相位归一化对齐 (跨节奏可比)
    loading_s = max((contact - swing_start) / fps, 1e-3)
    # 裁窗按 loading 成比例 (而非固定秒), 使不同节奏的人覆盖同一相位区间 ≈[-1.4, +0.6]
    pre_s = float(np.clip(1.4 * loading_s, 0.7, 3.0))
    post_s = float(np.clip(0.6 * loading_s, 0.4, 1.5))
    lo = max(0, contact - int(round(pre_s * fps)))
    hi = min(world.shape[0], contact + int(round(post_s * fps)) + 1)
    signals = _crop_signals(full, lo, hi)
    local_contact = contact - lo                        # 窗口内索引
    metrics = compute_metrics(signals, local_contact)
    metrics["contact_frame"] = int(contact)             # 覆盖为全段帧
    metrics["contact_t"] = float(contact / fps)
    # 发力链展开度归一化为相位 (节奏无关): 展开秒数 ÷ 装载时长
    metrics["seq_lead"] = round(metrics.pop("prox_lead_s", 0.0) / loading_s, 3)
    # 击球点 (身体坐标系, 视角无关): 持拍手腕前伸量 / 高度
    cfwd, chgt = contact_point(world[contact], full["hand"])
    metrics["contact_forward"] = cfwd
    metrics["contact_height"] = chgt
    return {"signals": signals, "contact": int(contact),
            "contact_local": int(local_contact), "world": world,
            "facing": facing, "n_frames": int(world.shape[0]),
            "swing_start": int(swing_start), "loading_s": float(loading_s),
            "metrics": metrics, "valid_ratio": float(valid.mean())}
