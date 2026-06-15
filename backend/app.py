"""
app.py — SmartTennis MVP 后端 (FastAPI)
========================================
POST /api/analyze  上传 10 秒正手视频 → 返回:
    - 动力链时序图 (PNG, base64)
    - 影子骨架叠加图 (PNG, base64)
    - 对照德约 IQR band 的诊断报告 (JSON)
GET  /            手机友好的前端页面
GET  /api/health  健康检查
"""
from __future__ import annotations
import base64, collections, json, os, tempfile, uuid

import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import pose
import kinetic_chain as kc
import diagnose
import shadow

BASE = os.path.dirname(__file__)
FRONTEND = os.path.abspath(os.path.join(BASE, "..", "frontend"))
REFERENCE_PATH = os.path.join(BASE, "reference", "djokovic_forehand.json")
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_MB", "60")) * 1024 * 1024

app = FastAPI(title="SmartTennis MVP")

with open(REFERENCE_PATH, encoding="utf-8") as fh:
    REFERENCE = json.load(fh)

# 德约参考的面朝方向 (用于跨机位镜像对齐)
_ref_cp = REFERENCE["reference"].get("contact_pose_img")
REF_FACING = kc.detect_facing([{"img": _ref_cp}]) if _ref_cp else 1.0


def _nearest_pose(frames, idx, radius=10):
    """从 idx 向两侧找最近一个检测到人体的帧 (避免击球糊帧没骨架)。"""
    n = len(frames)
    for d in range(radius + 1):
        for j in (idx - d, idx + d):
            if 0 <= j < n and frames[j].get("img"):
                return j, frames[j]["img"]
    return idx, None


# ── 会话缓存: 保存已提取的 landmarks 与视频, 供滑杆校正 contact 后快速重算 ──
SESSIONS: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
MAX_SESSIONS = 12


def _store_session(token: str, landmarks: dict, video_path: str, hand: str) -> None:
    SESSIONS[token] = {"landmarks": landmarks, "video": video_path, "hand": hand}
    while len(SESSIONS) > MAX_SESSIONS:           # 淘汰最旧会话并删其临时视频
        _, old = SESSIONS.popitem(last=False)
        try:
            os.unlink(old["video"])
        except OSError:
            pass


def _build_result(landmarks: dict, video_path: str, hand: str,
                  contact_override: int = None) -> dict:
    """跑分析 + 渲染, 给定 contact(自动或滑杆指定)。analyze/recompute 共用。"""
    res = kc.analyze(landmarks, hand=hand, contact_override=contact_override)
    report = diagnose.diagnose(res["metrics"], REFERENCE)
    ref = REFERENCE["reference"]
    chart = shadow.render_kinetic_chart(
        res["signals"], res["metrics"]["contact_t"], ref.get("ideal_curve"),
        user_loading_s=res.get("loading_s", 0.0))

    contact_idx, contact_pose = _nearest_pose(landmarks["frames"], res["contact"])
    user_contact = None
    frame = shadow.grab_frame(video_path, contact_idx)
    if frame is not None and contact_pose:
        user_contact = shadow.draw_skeleton_on_frame(frame, contact_pose)

    mirror_user = res.get("facing", 1.0) != REF_FACING
    overlay = None
    if contact_pose and ref.get("contact_pose_img"):
        overlay = shadow.render_shadow_overlay(
            contact_pose, ref["contact_pose_img"], mirror_user=mirror_user)

    scalar = {k: round(float(v), 3) for k, v in res["metrics"].items()
              if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {
        "ok": True, "hand": res["signals"]["hand"],
        "valid_ratio": round(res["valid_ratio"], 3),
        "contact": int(res["contact"]), "n_frames": int(res.get("n_frames") or 0),
        "metrics": scalar, "report": report,
        "kinetic_chart": chart, "shadow_overlay": overlay,
        "user_contact": user_contact,
        "djokovic_contact": "/assets/djokovic_contact.jpg",
    }


def _thumbs(video_path: str, lo: int, hi: int, width: int = 132) -> dict:
    """抠出 [lo,hi] 每帧的小缩略图(base64), 供前端滑杆即时预览击球帧。"""
    out = {}
    cap = cv2.VideoCapture(video_path)
    for idx in range(lo, hi + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        th = cv2.resize(fr, (width, max(1, int(h * width / w))))
        ok2, buf = cv2.imencode(".jpg", th, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok2:
            out[str(idx)] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    cap.release()
    return out


@app.get("/api/health")
def health():
    return {"status": "ok", "reference_strokes": REFERENCE.get("n_strokes")}


@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...), hand: str = Form("auto")):
    data = await video.read()
    if not data:
        raise HTTPException(400, "空文件")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"文件过大 (上限 {MAX_UPLOAD // 1024 // 1024}MB)")
    hand = hand if hand in ("L", "R") else "auto"   # 用户指定惯用手, 否则自动判定

    token = uuid.uuid4().hex
    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    path = os.path.join(tempfile.gettempdir(), f"st_{token}{suffix}")
    with open(path, "wb") as fh:
        fh.write(data)
    try:
        landmarks = pose.extract_from_video(path)
    except RuntimeError as e:
        os.unlink(path)
        raise HTTPException(422, str(e))

    _store_session(token, landmarks, path, hand)   # 保留视频供滑杆重算
    result = _build_result(landmarks, path, hand)

    # 滑杆候选窗口 (击球前后) + 逐帧缩略图, 供用户校正击球帧
    fps = float(landmarks.get("fps", 30.0))
    n, c = result["n_frames"], result["contact"]
    lo = max(0, c - int(round(0.8 * fps)))
    hi = min(n - 1, c + int(round(0.5 * fps)))
    result.update({"token": token, "window": [lo, hi],
                   "thumbs": _thumbs(path, lo, hi)})
    return JSONResponse(result)


@app.post("/api/recompute")
async def recompute(token: str = Form(...), contact: int = Form(...)):
    """滑杆校正击球帧后, 用缓存的 landmarks/视频快速重算 (不重跑 MediaPipe)。"""
    sess = SESSIONS.get(token)
    if not sess:
        raise HTTPException(404, "会话已过期, 请重新上传分析")
    SESSIONS.move_to_end(token)
    result = _build_result(sess["landmarks"], sess["video"], sess["hand"],
                           contact_override=int(contact))
    return JSONResponse(result)


# 前端静态资源 (放最后, 不覆盖 /api/*)
if os.path.isdir(FRONTEND):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND, "index.html"))
    app.mount("/", StaticFiles(directory=FRONTEND), name="static")
