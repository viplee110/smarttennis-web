"""
shadow.py — 服务端渲染动力链图 + 影子骨架叠加, 返回 base64 PNG
==============================================================
- render_kinetic_chart: 五环节归一化曲线 + X-factor, 可叠加德约理想曲线
- render_shadow_overlay: 用户 vs 德约 在 contact 帧的骨架, 归一化身材后叠加
"""
from __future__ import annotations
import base64, io
import numpy as np
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


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _use_cjk_font():
    """尽量挑一个能显示中文的字体, 没有则退回默认 (英文标签)。"""
    from matplotlib import font_manager
    for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "WenQuanYi Zen Hei", "Arial Unicode MS"]:
        if any(name.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["axes.unicode_minus"] = False
    return False


def render_kinetic_chart(signals: dict, contact_t: float,
                         ideal_curve: dict | None = None) -> str:
    cjk = _use_cjk_font()
    lab = {"hip": "Hip", "shoulder": "Shoulder", "upper_arm": "Upper arm",
           "forearm": "Forearm", "wrist": "Wrist"}
    t = np.asarray(signals["t"]) - float(contact_t)   # 相对击球时间, 击球=0
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 4.6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for key, zh, c in SIGNAL_STYLE:
        ax1.plot(t, signals["norm"][key], color=c, lw=1.8,
                 label=(zh if cjk else lab[key]))
    if ideal_curve:                                   # 德约理想曲线 (淡虚线, 已是相对时间)
        ti = np.asarray(ideal_curve["t"])
        for key, _zh, c in SIGNAL_STYLE:
            ax1.plot(ti, ideal_curve[key], color=c, lw=1.0, ls=":", alpha=0.5)
    ax1.axvline(0, ls="--", color="gray", lw=1)
    ax1.text(0, 1.14, ("击球" if cjk else "contact"),
             ha="center", fontsize=8, color="gray")
    ax1.set_ylabel("归一化角速度/速度" if cjk else "normalized speed", fontsize=9)
    title = ("动力链时序 (实线=你, 虚线=德约)" if cjk
             else "Kinetic chain (solid=you, dotted=Djokovic)")
    ax1.set_title(title, fontsize=10)
    ax1.legend(fontsize=7, ncol=3, loc="upper left")
    ax1.set_ylim(0, 1.2)
    ax2.plot(t, signals["xfactor"], color="#9467bd", lw=1.8)
    if ideal_curve and "xfactor" in ideal_curve:
        ax2.plot(np.asarray(ideal_curve["t"]), ideal_curve["xfactor"],
                 color="#9467bd", lw=1.0, ls=":", alpha=0.5)
    ax2.axvline(0, ls="--", color="gray", lw=1)
    ax2.set_ylabel("X-factor (°)", fontsize=9)
    ax2.set_xlabel("相对击球时间 (s)" if cjk else "time relative to contact (s)", fontsize=9)
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


def _draw_skeleton(ax, pts, color, alpha=1.0, lw=2.2):
    for a, b in CONNECTIONS:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    ax.scatter(pts[:, 0], pts[:, 1], s=10, color=color, alpha=alpha, zorder=3)


def render_shadow_overlay(user_pose_img, ref_pose_img,
                          user_hand: str = "R", ref_hand: str = "R") -> str:
    cjk = _use_cjk_font()
    mirror_user = user_hand != ref_hand              # 惯用手不同 → 镜像用户以可比
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
