"""
shadow.py — 服务端渲染动力链图 + 影子骨架叠加, 返回 base64 PNG
==============================================================
- render_kinetic_chart: 五环节归一化曲线 + X-factor, 可叠加德约理想曲线
- render_shadow_overlay: 用户 vs 德约 在 contact 帧的骨架, 归一化身材后叠加
"""
from __future__ import annotations
import base64, io
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# MediaPipe Pose 连接 (画干净的火柴人)
CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),          # 躯干
    (12, 14), (14, 16), (11, 13), (13, 15),          # 双臂
    (24, 26), (26, 28), (23, 25), (25, 27),          # 双腿
    (0, 11), (0, 12),                                # 头-肩
]
L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
SIGNAL_STYLE = [
    ("hip", "髋转动", "#2ca02c"), ("shoulder", "肩转动", "#1f77b4"),
    ("upper_arm", "上臂", "#9467bd"), ("forearm", "前臂", "#ff7f0e"),
    ("wrist", "手腕速度", "#d62728"),
]


def _fig_to_b64(fig, tight: bool = True) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches=("tight" if tight else None))
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# 发力时间轴的固定坐标轴(不裁剪), 让前端能按相位在图上精确画"当前帧"竖线。
# left/right = 绘图区占整张图宽度的比例; xmin/xmax = 相位轴范围。两者须与 render 一致。
SEQ_AXIS = {"xmin": -1.65, "xmax": 0.72, "left": 0.13, "right": 0.97}


# 字体名含这些标记之一即视为中文字体 (覆盖 Win/Linux/mac 常见中文字体)。
# 松散匹配 → 对 matplotlib 不同版本/.ttc 变体后缀的命名差异更健壮, 避免线上图表退回英文/豆腐块。
_CJK_MARKERS = ("yahei", "simhei", "simsun", "song", "noto sans cjk", "noto serif cjk",
                "source han", "wenquanyi", "wqy", "zenhei", "zen hei", "pingfang",
                "hiragino", "heiti", "kaiti", "fangsong", "droid sans fallback")


def _use_cjk_font():
    """挑一个能显示中文的字体, 没有则退回默认 (英文标签)。
    先按优先级找具名字体, 找不到再松散扫描任何含 CJK 标记的字体; 都用其"真实注册名"设置,
    避免候选名与实际名略有出入(如 .ttc 变体后缀)时 matplotlib 解析不到而仍渲染豆腐块。"""
    from matplotlib import font_manager
    fonts = font_manager.fontManager.ttflist

    def _apply(real_name: str) -> bool:
        plt.rcParams["font.sans-serif"] = [real_name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return True

    for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "WenQuanYi Zen Hei", "Arial Unicode MS"]:      # 1) 优先级具名匹配
        for f in fonts:
            if name.lower() in f.name.lower():
                return _apply(f.name)
    for f in fonts:                                             # 2) 兜底: 任意含 CJK 标记的字体
        if any(m in f.name.lower() for m in _CJK_MARKERS):
            return _apply(f.name)
    plt.rcParams["axes.unicode_minus"] = False
    return False


_SEG_STYLE = [("hip", "髋", "#2e7d32"), ("shoulder", "肩", "#1f77b4"),
              ("upper_arm", "上臂", "#9467bd"), ("forearm", "前臂", "#ff7f0e"),
              ("wrist", "手腕", "#d62728")]
# 无中文字体时的英文兜底标签 (比 k[:2]='hi/sh/up' 易懂)
_SEG_EN = {"hip": "Hip", "shoulder": "Shoulder", "upper_arm": "U.arm",
           "forearm": "Forearm", "wrist": "Wrist"}


def render_sequence_timeline(user_pt: dict, ref_pt: dict,
                             user_loading: float, ref_loading: float) -> str:
    """极简发力时间轴: 每个部位一个点(发力时刻, 相位), 你(上)/德约(下)两行。
    一眼看'谁先谁后、和德约差多少' —— 比5条交叠曲线直观得多。"""
    cjk = _use_cjk_font()
    du = user_loading if user_loading and user_loading > 1e-3 else 1.0
    dr = ref_loading if ref_loading and ref_loading > 1e-3 else 1.0
    xmin = SEQ_AXIS["xmin"]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    fig.subplots_adjust(left=SEQ_AXIS["left"], right=SEQ_AXIS["right"], top=0.80, bottom=0.20)
    rows = [(1.0, "你" if cjk else "You", user_pt, du),
            (0.0, "德约" if cjk else "Djokovic", ref_pt, dr)]
    # 各环节挤在 ~3 帧内(<0.10s)= 低于手机帧率分辨率, 精确先后是噪声 → 淡化用户5点、收成"一团",
    # 不当判决, 把结论交给抗噪指标(击球前旋转完成度 / 发力链顺序 组级)。
    uvals = [user_pt.get(k) for k, _zh, _c in _SEG_STYLE if user_pt.get(k) is not None]
    user_bunched = len(uvals) >= 3 and (max(uvals) - min(uvals)) < 0.10
    for y, name, pt, load in rows:
        if not pt:
            continue
        faded = (y == 1.0) and user_bunched
        a = 0.25 if faded else 1.0
        xs = [(pt.get(k, 0.0) / load) for k, _zh, _c in _SEG_STYLE]
        ax.plot(xs, [y] * len(xs), color="#cfd8d3", lw=2, zorder=1, alpha=a)   # 连线=发力链展开
        for (k, zh, c), x in zip(_SEG_STYLE, xs):
            ax.scatter([x], [y], s=150, color=c, zorder=3, edgecolors="white", linewidths=1.4, alpha=a)
        if faded:                                        # 淡化的点上盖一团 + 把结论引到抗噪指标
            cx = float(np.mean(xs))
            ax.scatter([cx], [y], s=1400, color="#c0392b", alpha=0.10, zorder=2)
            ax.text(cx, y - 0.34, "挤在一起·此帧率分不清精细先后 → 看下方『击球前旋转完成度』" if cjk
                    else "bunched (low fps) — see diagnosis below",
                    ha="center", va="top", fontsize=7.5, color="#c0392b", fontweight="bold")
        ax.text(xmin + 0.04, y, name, ha="left", va="center", fontsize=12, fontweight="bold")
    # 顶部色例 (替代逐点标签, 避免点挤时重叠)
    for i, (k, zh, c) in enumerate(_SEG_STYLE):
        x0 = -1.4 + i * 0.42
        ax.scatter([x0], [1.62], s=70, color=c, edgecolors="white", linewidths=1)
        ax.text(x0 + 0.05, 1.62, zh if cjk else _SEG_EN.get(k, k), va="center", fontsize=9, color="#333")
    ax.axvline(0, ls="--", color="gray", lw=1)
    ax.text(xmin + 0.04, -0.62, "← 越靠左=越早发力；点拉得越开=发力链越依次展开(德约式)" if cjk
            else "← earlier; more spread = better chain", ha="left", fontsize=8.5, color="#888")
    ax.set_xlim(xmin, SEQ_AXIS["xmax"]); ax.set_ylim(-0.8, 1.8)
    ax.set_yticks([]); ax.set_xlabel("挥拍相位 (0=击球)" if cjk else "swing phase (0=contact)", fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return _fig_to_b64(fig, tight=False)   # 固定边距 → 前端可精确定位竖线


def render_kinetic_chart(signals: dict, contact_t: float,
                         ideal_curve: dict | None = None,
                         user_loading_s: float = 0.0) -> str:
    """相位归一化对齐: 横轴 = 挥拍相位 (0=击球, -1=前挥起点)。
    各自除以自己的"装载时长", 让两人在【前挥起点 + 击球】两点对齐, 跨节奏可比。
    """
    cjk = _use_cjk_font()
    lab = {"hip": "Hip", "shoulder": "Shoulder", "upper_arm": "Upper arm",
           "forearm": "Forearm", "wrist": "Wrist"}
    du = float(user_loading_s) if user_loading_s and user_loading_s > 1e-3 else 1.0
    t = (np.asarray(signals["t"]) - float(contact_t)) / du   # 归一化相位, 击球=0
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 4.6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for key, zh, c in SIGNAL_STYLE:
        ax1.plot(t, signals["norm"][key], color=c, lw=1.8,
                 label=(zh if cjk else lab[key]))
    if ideal_curve:                                   # 德约理想曲线 (淡虚线), 同样按相位归一化
        dj = float(ideal_curve.get("loading_s") or 0.0)
        dj = dj if dj > 1e-3 else 1.0
        ti = np.asarray(ideal_curve["t"]) / dj
        for key, _zh, c in SIGNAL_STYLE:
            ax1.plot(ti, ideal_curve[key], color=c, lw=1.0, ls=":", alpha=0.5)
    for ax in (ax1, ax2):
        ax.axvline(0, ls="--", color="gray", lw=1)        # 击球
        ax.axvline(-1, ls=":", color="#bbb", lw=1)        # 前挥起点
    ax1.text(0, 1.14, ("击球" if cjk else "contact"), ha="center", fontsize=8, color="gray")
    ax1.text(-1, 1.14, ("前挥起点" if cjk else "swing start"), ha="center", fontsize=7.5, color="#aaa")
    ax1.set_ylabel("归一化角速度/速度" if cjk else "normalized speed", fontsize=9)
    title = ("动力链时序 (实线=你, 虚线=德约; 已相位对齐)" if cjk
             else "Kinetic chain (solid=you, dotted=Djokovic; phase-aligned)")
    ax1.set_title(title, fontsize=10)
    ax1.legend(fontsize=7, ncol=3, loc="upper left")
    ax1.set_ylim(0, 1.2)
    ax1.set_xlim(-1.6, 0.9)
    ax2.plot(t, signals["xfactor"], color="#9467bd", lw=1.8)
    if ideal_curve and "xfactor" in ideal_curve:
        dj = float(ideal_curve.get("loading_s") or 0.0)
        dj = dj if dj > 1e-3 else 1.0
        ax2.plot(np.asarray(ideal_curve["t"]) / dj, ideal_curve["xfactor"],
                 color="#9467bd", lw=1.0, ls=":", alpha=0.5)
    ax2.set_ylabel("X-factor (°)", fontsize=9)
    ax2.set_xlabel("挥拍相位 (0=击球, -1=前挥起点)" if cjk
                   else "swing phase (0=contact, -1=swing start)", fontsize=9)
    return _fig_to_b64(fig)


def _normalize_skeleton(pose_img, mirror: bool = False) -> np.ndarray:
    """img 坐标(33×2, y向下) → 以髋中心为原点、躯干长为单位、y向上 的骨架。"""
    p = np.array(pose_img, dtype=float)[:, :2]
    p[:, 1] = -p[:, 1]                                # y 翻成向上
    if mirror:
        p[:, 0] = -p[:, 0]
    hip_c = (p[L_HIP] + p[R_HIP]) / 2
    sh_c = (p[L_SH] + p[R_SH]) / 2
    scale = np.linalg.norm(sh_c - hip_c) or 1e-6
    return (p - hip_c) / scale


def _project_canonical_sideview(world33) -> np.ndarray:
    """world 3D 关节点 → 统一侧视(矢状面)的 2D 骨架, 与拍摄机位无关。

    用解剖坐标系(右髋-左髋=左右轴, 髋→肩=躯干向上)推出'前方'=cross(右,上),
    再投影到 (前方, 上) 平面 → 不管原视频是正侧/斜前拍, 都还原成同一虚拟侧视相机,
    从根本上消除'骨架宽度不一致'。以躯干长归一化大小。
    """
    p = np.array(world33, dtype=float)[:, :3]
    hip_c = (p[L_HIP] + p[R_HIP]) / 2.0
    sh_c = (p[L_SH] + p[R_SH]) / 2.0
    p = p - hip_c
    right = p[R_HIP] - p[L_HIP]
    right = right / (np.linalg.norm(right) + 1e-9)
    up = sh_c - hip_c
    up = up - np.dot(up, right) * right          # 对 right 正交化
    up = up / (np.linalg.norm(up) + 1e-9)
    forward = np.cross(right, up)
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    xy = np.stack([p @ forward, p @ up], axis=1)  # x=前后, y=上下
    scale = np.linalg.norm(sh_c - hip_c) or 1e-6
    return xy / scale


def _draw_skeleton(ax, pts, color, alpha=1.0, lw=2.2):
    for a, b in CONNECTIONS:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    ax.scatter(pts[:, 0], pts[:, 1], s=10, color=color, alpha=alpha, zorder=3)


def draw_skeleton_on_frame(frame_bgr: np.ndarray, pose_img, max_w: int = 900) -> str:
    """把 2D 骨架画在真实视频帧上, 返回 base64 JPEG (直观显示击球瞬间)。
    pose_img: 33×(x,y,...) 归一化坐标 [0,1]。"""
    h, w = frame_bgr.shape[:2]
    p = np.array(pose_img, dtype=float)[:, :2] * [w, h]
    lw = max(2, w // 350)
    r = max(3, w // 240)
    for a, b in CONNECTIONS:
        pa, pb = p[a].astype(int), p[b].astype(int)
        cv2.line(frame_bgr, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                 (0, 215, 255), lw, cv2.LINE_AA)
    for x, y in p.astype(int):
        cv2.circle(frame_bgr, (int(x), int(y)), r, (0, 120, 255), -1, cv2.LINE_AA)
    if w > max_w:
        frame_bgr = cv2.resize(frame_bgr, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# 同步 scrubber 的统一相位采样点 (τ: 0=击球, -1≈前挥起点)。两人共用 → 帧一一对应。
# 必须精确包含 τ=0(索引13): 使 scrubber 的"击球"帧 == 静态击球图帧 == REF_CONTACT, 永远一致。
SCRUB_PHASES = [round(-0.975 + i * 0.075, 3) for i in range(21)]   # -0.975..+0.525, 0 在索引13


def _nearest_img(frames_lm, fi, n):
    pose_img = frames_lm[fi].get("img")
    if not pose_img:
        for d in range(1, 8):
            for j in (fi - d, fi + d):
                if 0 <= j < n and frames_lm[j].get("img"):
                    return frames_lm[j]["img"]
    return pose_img


def scrub_strip_at(video_path: str, frames_lm, frame_indices, width: int = 240) -> list:
    """按给定帧索引列表生成 [真实帧+骨架] 缩略图序列(base64)。"""
    n = len(frames_lm)
    cap = cv2.VideoCapture(video_path)
    out = []
    for fi in frame_indices:
        fi = max(0, min(n - 1, int(fi)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            out.append(None)
            continue
        pose_img = _nearest_img(frames_lm, fi, n)
        out.append(draw_skeleton_on_frame(fr, pose_img, max_w=width) if pose_img else None)
    cap.release()
    return out


def scrub_strip(video_path: str, frames_lm, contact: int, loading_s: float,
                fps: float, width: int = 240, bounds=None) -> list:
    """沿统一相位轴(SCRUB_PHASES, 线性)生成帧条。德约预生成静态条用它。
    bounds=(lo,hi) 时把帧钳制在用户所选挥拍片段内, 不越界到片段外(随挥侧尤甚)。"""
    idx = [int(round(contact + tau * loading_s * fps)) for tau in SCRUB_PHASES]
    if bounds:
        lo, hi = int(bounds[0]), int(bounds[1])
        idx = [max(lo, min(hi, i)) for i in idx]
    return scrub_strip_at(video_path, frames_lm, idx, width)


def grab_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    """读取视频指定帧 (BGR)。"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def render_shadow_overlay(user_pose_img, ref_pose_img, mirror_user: bool = False,
                          user_world=None, ref_world=None) -> str:
    cjk = _use_cjk_font()
    # 优先用 3D world 做"视角归一化"侧视投影 (两人还原到同一虚拟相机, 消除机位差);
    # 缺 3D 时回退到 2D 图像坐标(受拍摄角度影响)。
    if user_world is not None and ref_world is not None:
        user = _project_canonical_sideview(user_world)
        ref = _project_canonical_sideview(ref_world)
    else:
        user = _normalize_skeleton(user_pose_img, mirror=mirror_user)
        ref = _normalize_skeleton(ref_pose_img)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.4))
    titles = (["你 (击球瞬间)", "德约 (击球瞬间)", "叠加对比"] if cjk
              else ["You (contact)", "Djokovic (contact)", "Overlay"])
    _draw_skeleton(axes[0], user, "#1f77b4")
    _draw_skeleton(axes[1], ref, "#ff7f0e")
    _draw_skeleton(axes[2], ref, "#ff7f0e", alpha=0.55, lw=2.0)
    _draw_skeleton(axes[2], user, "#1f77b4", alpha=0.95)
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-2.0, 1.4)
    return _fig_to_b64(fig)
