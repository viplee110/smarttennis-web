"""
pose.py — 用 MediaPipe 从视频逐帧提取 33 关节点 (复用已验证的提取逻辑)
=====================================================================
返回与 extract_landmarks.py 相同结构的 dict, 直接喂给 kinetic_chain.analyze。
"""
from __future__ import annotations
import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
)

MODEL_PATH = os.environ.get(
    "POSE_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_lite.task"),
)
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "450"))   # ~15s@30fps 上限, 防滥用


def extract_from_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    frames, idx = [], 0
    try:
        while idx < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(img, int(idx * 1000.0 / fps))
            entry = {"frame": idx, "t": idx / fps}
            if res.pose_landmarks and res.pose_world_landmarks:
                lm, wm = res.pose_landmarks[0], res.pose_world_landmarks[0]
                entry["img"] = [[round(p.x, 5), round(p.y, 5), round(p.z, 5),
                                 round(p.visibility, 3)] for p in lm]
                entry["world"] = [[round(p.x, 5), round(p.y, 5), round(p.z, 5)]
                                  for p in wm]
            else:
                entry["img"] = entry["world"] = None
            frames.append(entry)
            idx += 1
    finally:
        cap.release()
        landmarker.close()

    n_ok = sum(1 for f in frames if f["world"])
    if idx == 0:
        raise RuntimeError("视频没有可读帧")
    if n_ok / idx < 0.5:
        raise RuntimeError(f"只有 {n_ok}/{idx} 帧检测到人体, 请确保画面中只有一位球员且全身可见")
    return {"video": os.path.basename(video_path), "fps": fps, "frames": frames}
