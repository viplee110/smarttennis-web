"""
mine_strokes.py — 从一段长视频一遍 MediaPipe 直接挖出每条挥拍的 landmark JSON
==============================================================================
替代 "smart_cutter 切片 + extract_landmarks 再提取" 两遍流程 (且不依赖 moviepy)。
按手腕图像速度峰检测挥拍, 每个峰前后各取一段窗口, 存成 build_reference 能直接
glob 的 <name>__stroke_NNN.json ({video, fps, frames:[{img,world}]})。

用法:
    python mine_strokes.py --video "<path.mp4>" --out "<landmarks_dir>" \
                           --model "models/pose_landmarker_lite.task"
"""
from __future__ import annotations
import argparse, json, os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
)

R_WR = 16            # 右手腕 (持拍手, 与既有德约数据一致)
PRE_S, POST_S = 1.5, 1.5      # 挥拍峰前后各取的秒数 (与 smart_cutter 一致)
THRESH_FRAC = 0.45            # 峰值需 ≥ 此比例 * 全局最大速度
MIN_GAP_S = 1.8               # 两次挥拍最小间隔


def _smooth(x, win=5):
    k = np.ones(win) / win
    return np.convolve(np.pad(x, win // 2, mode="reflect"), k, mode="valid")


def extract_all(video_path: str, model: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    lm = PoseLandmarker.create_from_options(opts)
    frames, idx = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                  int(idx * 1000.0 / fps))
        e = {"frame": idx, "t": idx / fps}
        if res.pose_landmarks and res.pose_world_landmarks:
            p, w = res.pose_landmarks[0], res.pose_world_landmarks[0]
            e["img"] = [[round(q.x, 5), round(q.y, 5), round(q.z, 5), round(q.visibility, 3)] for q in p]
            e["world"] = [[round(q.x, 5), round(q.y, 5), round(q.z, 5)] for q in w]
        else:
            e["img"] = e["world"] = None
        frames.append(e)
        idx += 1
    cap.release(); lm.close()
    return fps, frames


def find_peaks(frames, fps):
    T = len(frames)
    wx = np.array([f["img"][R_WR][0] if f.get("img") else np.nan for f in frames])
    wy = np.array([f["img"][R_WR][1] if f.get("img") else np.nan for f in frames])
    g = np.arange(T)
    for a in (wx, wy):
        m = ~np.isnan(a)
        if m.sum() < 2:
            return []
        a[~m] = np.interp(g[~m], g[m], a[m])
    spd = _smooth(np.hypot(np.gradient(wx), np.gradient(wy)))
    # 抗离群: 压制场景切换/检测跳变造成的尖峰, 再用分位数定阈值
    cap = np.percentile(spd, 98)
    spd = np.minimum(spd, cap)
    thr = max(np.percentile(spd, 70), cap * 0.45)
    gap = int(MIN_GAP_S * fps)
    cand = [i for i in range(1, T - 1)
            if spd[i] >= thr and spd[i] >= spd[i - 1] and spd[i] >= spd[i + 1]]
    peaks = []
    for i in cand:
        if not peaks or i - peaks[-1] >= gap:
            peaks.append(i)
        elif spd[i] > spd[peaks[-1]]:
            peaks[-1] = i
    print(f"  [peaks] thr={thr:.4f} cap={cap:.4f} cand={len(cand)} merged={len(peaks)}")
    return peaks


def main(video, outdir, model, cache=None):
    name = os.path.splitext(os.path.basename(video))[0]
    if cache and os.path.exists(cache):
        d = json.load(open(cache, encoding="utf-8"))
        fps, frames = d["fps"], d["frames"]
        print(f"  [cache] loaded {len(frames)} frames from {cache}")
    else:
        fps, frames = extract_all(video, model)
        if cache:
            os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
            json.dump({"fps": fps, "frames": frames}, open(cache, "w", encoding="utf-8"))
            print(f"  [cache] saved {len(frames)} frames -> {cache}")
    n_ok = sum(1 for f in frames if f["world"])
    peaks = find_peaks(frames, fps)
    pre, post = int(PRE_S * fps), int(POST_S * fps)
    os.makedirs(outdir, exist_ok=True)
    saved = 0
    for k, pk in enumerate(peaks, 1):
        lo, hi = max(0, pk - pre), min(len(frames), pk + post)
        sub = [dict(f) for f in frames[lo:hi]]
        if len(sub) < 30:
            continue
        out = {"video": f"{name}_stroke_{k:03d}.mp4", "fps": fps, "frames": sub}
        path = os.path.join(outdir, f"{name}__stroke_{k:03d}.json")
        json.dump(out, open(path, "w", encoding="utf-8"))
        saved += 1
    print(f"{name}: {len(frames)} frames ({n_ok} detected), {len(peaks)} peaks -> {saved} strokes")
    print("MINE_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__),
                                                    "models", "pose_landmarker_lite.task"))
    ap.add_argument("--cache", default=None, help="全帧缓存json, 复用避免重跑mediapipe")
    a = ap.parse_args()
    main(a.video, a.out, a.model, a.cache)
