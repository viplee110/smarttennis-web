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
MAX_DURATION_S = int(os.environ.get("MAX_DURATION_S", "30"))   # 上传视频时长上限(秒)

app = FastAPI(title="SmartTennis MVP")

with open(REFERENCE_PATH, encoding="utf-8") as fh:
    REFERENCE = json.load(fh)

# 德约参考的面朝方向 (用于跨机位镜像对齐)
_ref_cp = REFERENCE["reference"].get("contact_pose_img")
REF_FACING = kc.detect_facing([{"img": _ref_cp}]) if _ref_cp else 1.0


def _asset_ver() -> str:
    """德约静态图内容哈希, 作为资源URL版本号 → 图一变就强制浏览器重新下载,
    根治'重新生成了同名图但手机Safari仍显示旧缓存'的问题。"""
    import hashlib
    try:
        with open(os.path.join(FRONTEND, "assets", "djokovic_contact.jpg"), "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()[:8]
    except OSError:
        return "1"


ASSET_VER = _asset_ver()


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
                  contact_override: int = None, seg: tuple = None) -> dict:
    """跑分析 + 渲染, 给定 contact(自动或滑杆指定) 与可选挥拍片段 seg。analyze/recompute 共用。"""
    res = kc.analyze(landmarks, hand=hand, contact_override=contact_override, seg=seg)
    report = diagnose.diagnose(res["metrics"], REFERENCE)
    ref = REFERENCE["reference"]
    chart = shadow.render_kinetic_chart(
        res["signals"], res["metrics"]["contact_t"], ref.get("ideal_curve"),
        user_loading_s=res.get("loading_s", 0.0))
    # 极简发力时间轴 (主视觉, 比5曲线直观)
    seq_chart = shadow.render_sequence_timeline(
        res["metrics"].get("peak_times") or {}, ref.get("peak_times") or {},
        res.get("loading_s", 1.0), (ref.get("ideal_curve") or {}).get("loading_s", 1.0))

    contact_idx, contact_pose = _nearest_pose(landmarks["frames"], res["contact"])
    user_contact = None
    frame = shadow.grab_frame(video_path, contact_idx)
    if frame is not None and contact_pose:
        user_contact = shadow.draw_skeleton_on_frame(frame, contact_pose)

    mirror_user = res.get("facing", 1.0) != REF_FACING
    # 3D 视角归一化: 取用户/德约 contact 帧的 world 坐标, 还原同一侧视
    uw = res["world"][contact_idx].tolist() if 0 <= contact_idx < len(res["world"]) else None
    overlay = None
    if contact_pose and ref.get("contact_pose_img"):
        overlay = shadow.render_shadow_overlay(
            contact_pose, ref["contact_pose_img"], mirror_user=mirror_user,
            user_world=uw, ref_world=ref.get("contact_pose_world"))

    # 同步逐帧 scrubber: 线性相位映射(每相位点=各自 contact + τ·loading)。
    # 注: 曾试 DTW 非线性对应, 但它会按信号相似度 warp 掉用户的击球帧, 致逐帧不同步,
    # 反而忠实度更差 → 回退线性, 让用户能扫到自己挥拍的每一帧。
    fps_u = float(landmarks.get("fps", 30.0))
    scrub_user = shadow.scrub_strip(
        video_path, landmarks["frames"], res["contact"], res.get("loading_s", 1.0), fps_u)

    scalar = {k: round(float(v), 3) for k, v in res["metrics"].items()
              if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {
        "ok": True, "hand": res["signals"]["hand"],
        "valid_ratio": round(res["valid_ratio"], 3),
        "contact": int(res["contact"]), "n_frames": int(res.get("n_frames") or 0),
        "metrics": scalar, "report": report,
        "kinetic_chart": chart, "sequence_chart": seq_chart, "seq_axis": shadow.SEQ_AXIS,
        "shadow_overlay": overlay,
        "user_contact": user_contact,
        "djokovic_contact": f"/assets/djokovic_contact.jpg?v={ASSET_VER}",
        "scrub_user": scrub_user,
        "scrub_djoko": [f"/assets/djoko_scrub/{i:02d}.jpg?v={ASSET_VER}"
                        for i in range(len(shadow.SCRUB_PHASES))],
        "scrub_phases": shadow.SCRUB_PHASES,
    }


def _encode_thumb(fr, width: int = 132) -> "str | None":
    h, w = fr.shape[:2]
    th = cv2.resize(fr, (width, max(1, int(h * width / w))))
    ok, buf = cv2.imencode(".jpg", th, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _frame_thumbs(video_path: str, indices, width: int = 132) -> dict:
    """抠出指定帧号的小缩略图(base64), 返回 {帧号字符串: dataURL}。"""
    out = {}
    cap = cv2.VideoCapture(video_path)
    for idx in indices:
        idx = int(idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok:
            t = _encode_thumb(fr, width)
            if t:
                out[str(idx)] = t
    cap.release()
    return out


def _thumbs(video_path: str, lo: int, hi: int, width: int = 132) -> dict:
    """[lo,hi] 每帧缩略图, 供击球帧滑杆即时预览。"""
    return _frame_thumbs(video_path, range(lo, hi + 1), width)


def _filmstrip(video_path: str, n_frames: int, fps: float, max_thumbs: int = 60) -> dict:
    """全视频稀疏缩略图(约每 0.4s 一帧), 供"片段起止"滑杆拖动时预览。"""
    if n_frames <= 0:
        return {}
    step = max(int(round(0.4 * fps)), (n_frames + max_thumbs - 1) // max_thumbs, 1)
    idxs = list(range(0, n_frames, step))
    if idxs and idxs[-1] != n_frames - 1:
        idxs.append(n_frames - 1)
    return _frame_thumbs(video_path, idxs, width=120)


def _contact_slider(video_path: str, fps: float, contact: int, n: int, bounds=None):
    """击球帧校正滑杆的窗口 [lo,hi] 及逐帧缩略图; bounds 给定时钳制在挥拍片段内。"""
    lo = max(0, contact - int(round(0.8 * fps)))
    hi = min(n - 1, contact + int(round(0.5 * fps)))
    if bounds:
        lo = max(lo, int(bounds[0])); hi = min(hi, int(bounds[1]))
    if hi < lo:
        hi = lo
    return [lo, hi], _thumbs(video_path, lo, hi)


def _video_duration(video_path: str):
    """快速读视频时长(秒); 拿不到返回 None。"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nf = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return (nf / fps) if fps > 0 and nf > 0 else None


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

    # 时长上限预检 (避免在超长视频上空跑 MediaPipe)
    dur = _video_duration(path)
    if dur is not None and dur > MAX_DURATION_S + 1.0:
        os.unlink(path)
        raise HTTPException(413, f"视频时长约 {dur:.0f} 秒, 超过 {MAX_DURATION_S} 秒上限。"
                                 f"请先裁剪到只含挥拍的 {MAX_DURATION_S} 秒内再上传。")
    try:
        landmarks = pose.extract_from_video(path)
    except RuntimeError as e:
        os.unlink(path)
        raise HTTPException(422, str(e))

    frames = landmarks["frames"]
    fps = float(landmarks.get("fps", 30.0))
    n = len(frames)
    if n < 10:                                      # 帧太少 → 无法构成一次挥拍 (也防下游空数组)
        os.unlink(path)
        raise HTTPException(422, "视频太短或未能稳定提取到骨架, 请上传包含完整挥拍(引拍→收拍)的视频。")
    if n / fps > MAX_DURATION_S + 1.0:              # 后备: 部分容器读不到时长元数据
        os.unlink(path)
        raise HTTPException(413, f"视频时长约 {n / fps:.0f} 秒, 超过 {MAX_DURATION_S} 秒上限, 请裁剪后再传。")

    _store_session(token, landmarks, path, hand)    # 保留视频供片段/滑杆重算

    # 自动切出候选挥拍片段 (纯姿态, 无重型 ML); 默认分析最强的一拍, 其余供用户切换
    facing = kc.detect_facing(frames)
    real_hand = hand if hand in ("L", "R") else kc.detect_handedness(kc.load_world(landmarks)[0], fps)
    swings = kc.detect_swing_multiple(frames, fps, real_hand, facing)
    default_seg = None
    if swings:
        best = max(swings, key=lambda s: s["vf"])
        default_seg = (best["swing_start"], best["end"])

    result = _build_result(landmarks, path, hand, seg=default_seg)
    c = result["contact"]
    win, thumbs = _contact_slider(path, fps, c, n, default_seg)
    cand_thumbs = _frame_thumbs(path, [s["contact"] for s in swings])
    result.update({
        "token": token, "n_frames": n, "fps": fps,
        "window": win, "thumbs": thumbs,
        "filmstrip": _filmstrip(path, n, fps),
        "swings": [{"i": k, "swing_start": s["swing_start"], "contact": s["contact"],
                    "end": s["end"], "duration_s": round((s["end"] - s["swing_start"]) / fps, 2),
                    "thumb": cand_thumbs.get(str(s["contact"]))}
                   for k, s in enumerate(swings)],
    })
    return JSONResponse(result)


@app.post("/api/recompute")
async def recompute(token: str = Form(...), contact: int = Form(-1),
                    seg_lo: int = Form(-1), seg_hi: int = Form(-1)):
    """滑杆校正击球帧 / 选定挥拍片段后, 用缓存 landmarks 快速重算 (不重跑 MediaPipe)。
    contact<0 表示该片段内自动找击球; seg_lo/seg_hi 给定时只在该片段内分析。"""
    sess = SESSIONS.get(token)
    if not sess:
        raise HTTPException(404, "会话已过期, 请重新上传分析")
    SESSIONS.move_to_end(token)
    landmarks = sess["landmarks"]
    fps = float(landmarks.get("fps", 30.0))
    n = len(landmarks["frames"])
    seg = (seg_lo, seg_hi) if (seg_lo >= 0 and seg_hi > seg_lo) else None
    contact_override = int(contact) if contact >= 0 else None
    result = _build_result(landmarks, sess["video"], sess["hand"],
                           contact_override=contact_override, seg=seg)
    c = result["contact"]
    win, thumbs = _contact_slider(sess["video"], fps, c, n, seg)
    result.update({"token": token, "n_frames": n, "fps": fps,
                   "window": win, "thumbs": thumbs})
    return JSONResponse(result)


# 前端静态资源 (放最后, 不覆盖 /api/*)
if os.path.isdir(FRONTEND):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND, "index.html"))
    app.mount("/", StaticFiles(directory=FRONTEND), name="static")
