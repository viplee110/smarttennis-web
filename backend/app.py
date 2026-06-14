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
import json, os, tempfile

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

    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data); tmp.close()
    try:
        try:
            landmarks = pose.extract_from_video(tmp.name)
        except RuntimeError as e:
            raise HTTPException(422, str(e))

        res = kc.analyze(landmarks, hand=hand)
        report = diagnose.diagnose(res["metrics"], REFERENCE)
        ref = REFERENCE["reference"]
        chart = shadow.render_kinetic_chart(
            res["signals"], res["metrics"]["contact_t"], ref.get("ideal_curve"))

        contact_idx = res["contact"]
        contact_pose = landmarks["frames"][contact_idx].get("img")

        # 用户击球瞬间真实帧 + 骨架
        user_contact = None
        frame = shadow.grab_frame(tmp.name, contact_idx)
        if frame is not None and contact_pose:
            user_contact = shadow.draw_skeleton_on_frame(frame, contact_pose)

        overlay = None
        if contact_pose and ref.get("contact_pose_img"):
            overlay = shadow.render_shadow_overlay(
                contact_pose, ref["contact_pose_img"],
                user_hand=res["signals"]["hand"], ref_hand=ref.get("hand", "R"))

        return JSONResponse({
            "ok": True,
            "hand": res["signals"]["hand"],
            "valid_ratio": round(res["valid_ratio"], 3),
            "metrics": {k: round(float(v), 3) for k, v in res["metrics"].items()},
            "report": report,
            "kinetic_chart": chart,
            "shadow_overlay": overlay,
            "user_contact": user_contact,
            "djokovic_contact": "/assets/djokovic_contact.jpg",
        })
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# 前端静态资源 (放最后, 不覆盖 /api/*)
if os.path.isdir(FRONTEND):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND, "index.html"))
    app.mount("/", StaticFiles(directory=FRONTEND), name="static")
