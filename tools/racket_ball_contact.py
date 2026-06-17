"""
racket_ball_contact.py — 【离线/可选】用球拍+球检测精确定位真·击球帧
====================================================================
⚠️ 这是离线工具, 刻意不进主部署 (FastAPI 镜像)。它依赖重型 ML(torch/ultralytics),
   会让镜像涨到 GB 级, 不适合小 VPS。请在有 GPU/较强 CPU 的机器上单独安装运行:

       pip install ultralytics            # 自带 torch; 仅在此机器装, 不要加进 backend/requirements.txt

用途: 姿态只给手腕, 球碰的是拍面(远手腕~0.5m), 故纯姿态的 contact 有 ~±0.4s 残差。
      球拍+球检测能把"真·拍面触球帧"精确到 1-2 帧。
      推荐先用它**离线精修固定的德约参考 contact**(参考是固定的, 值得一次性弄准),
      产出一个帧号, 再喂给 build_reference 的 REF_CONTACT / analyze 的 contact_override。
      用户侧暂仍用 姿态+滑杆; 等有更强服务器或离线批处理再上用户侧。

算法:
  1. YOLO(COCO) 逐帧检测 'tennis racket'(cls 38) 与 'sports ball'(cls 32)。
  2. 取每帧最高置信的拍/球中心; 球缺帧用前后插值(TrackNet 可显著提升小快糊球的召回, 见下注)。
  3. 真·contact = 球到拍中心距离的局部极小 且 球水平速度方向反转(被击回)那一帧。
  4. 输出 contact 帧号 + 置信度。

注: 'sports ball'(YOLO/COCO)对网球这种又小又快又糊的目标召回有限。更专业的是 **TrackNet**
    (热力图、堆叠3帧、专为小快糊球设计, 清晰视频上精度~99%)。本脚本先用 YOLO 打通流程,
    要更高精度时把第2步的球检测换成 TrackNet。

用法:
    python racket_ball_contact.py --video path.mp4 [--model yolov8n.pt] [--pose-contact 137]
"""
from __future__ import annotations
import argparse

RACKET_CLS = 38      # COCO 'tennis racket'
BALL_CLS = 32        # COCO 'sports ball'


def detect_tracks(video_path: str, model_name: str = "yolov8n.pt"):
    """逐帧检测, 返回 (racket_centers, ball_centers) 两个 {frame: (x,y)} 字典。"""
    import cv2
    from ultralytics import YOLO      # 重型依赖, 仅离线机器装
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)
    rackets, balls = {}, {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, verbose=False, classes=[RACKET_CLS, BALL_CLS])[0]
        best = {}                      # cls -> (conf, cx, cy)
        for b in res.boxes:
            c = int(b.cls[0]); conf = float(b.conf[0])
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if c not in best or conf > best[c][0]:
                best[c] = (conf, cx, cy)
        if RACKET_CLS in best:
            rackets[idx] = best[RACKET_CLS][1:]
        if BALL_CLS in best:
            balls[idx] = best[BALL_CLS][1:]
        idx += 1
    cap.release()
    return rackets, balls, idx


def find_contact(rackets: dict, balls: dict, n_frames: int):
    """真·contact = 球到拍距离极小 且 球水平速度反向 的帧。"""
    import numpy as np
    if not rackets or not balls:
        return None, 0.0
    f = np.arange(n_frames)
    def interp(d, axis):
        ks = sorted(d); xs = [d[k][axis] for k in ks]
        if len(ks) < 2:
            return None
        return np.interp(f, ks, xs)
    rx, ry = interp(rackets, 0), interp(rackets, 1)
    bx, by = interp(balls, 0), interp(balls, 1)
    if any(v is None for v in (rx, ry, bx, by)):
        return None, 0.0
    dist = np.hypot(bx - rx, by - ry)
    vbx = np.gradient(bx)                       # 球水平速度
    # 候选: 距离局部极小处; 在其中找速度反向(sign flip)的
    cands = [i for i in range(2, n_frames - 2)
             if dist[i] <= dist[i - 1] and dist[i] <= dist[i + 1]]
    best_i, best_score = None, 1e18
    for i in cands:
        flip = np.sign(vbx[max(0, i - 2)]) != np.sign(vbx[min(n_frames - 1, i + 2)])
        score = dist[i] - (50.0 if flip else 0.0)   # 反向给奖励
        if score < best_score:
            best_score, best_i = score, i
    conf = float(np.clip(1.0 - dist[best_i] / (np.median(dist) + 1e-6), 0, 1)) if best_i else 0.0
    return best_i, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--pose-contact", type=int, default=None, help="姿态法的contact帧, 用于对比")
    a = ap.parse_args()
    rackets, balls, n = detect_tracks(a.video, a.model)
    print(f"帧数 {n}; 检到球拍 {len(rackets)} 帧, 球 {len(balls)} 帧")
    cf, conf = find_contact(rackets, balls, n)
    print(f"真·contact 帧 = {cf}  (置信 {conf:.2f})")
    if a.pose_contact is not None and cf is not None:
        print(f"姿态法 contact = {a.pose_contact}, 差 {cf - a.pose_contact} 帧")
    print("→ 把这个帧号填入 build_reference.REF_CONTACT 或 analyze(contact_override=...)。")


if __name__ == "__main__":
    main()
