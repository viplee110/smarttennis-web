"""
coach.py — 教练盲评门户(效度评测数据采集)
=============================================
设计(2026-07-08):
- 一人一码: coach_data/codes.json {"码": {"name": "教练A"}}; 评分按码落盘, 天然归属。
- 盲评: 教练只看到原始视频片段(无骨架/无系统分数/无来源标注); 每位教练看同一批、
  但顺序按其授权码做确定性洗牌(防串答案时对照位置)。
- 评测集: coach_data/manifest.json [{"id":"c01","file":"clips/c01.mp4", ...隐藏字段}] ——
  隐藏字段(来源/锚点/重复标记)永不下发给前端, 只在离线分析时用。
- 数据: coach_data/ratings/<码>.json, 原子写盘; 容器外挂载(docker -v), 重部署不丢。
- 防不认真: 前端上报每条停留秒数; 重复/锚点片段在 manifest 里定义, 离线核查。

普通用户完全无感: 独立页面 /coach, 主页无入口; 不触碰任何分析代码路径。
"""
from __future__ import annotations
import json
import os
import random
import time

import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/coach")
_LOCK = threading.Lock()          # 评分文件读-改-写互斥(同码双开设备的竞态保护)

# 原因芯片白名单(教练行话, 刻意不含我们的指标名——离线才映射回指标, 免得给教练发答案)。
# 前 7 个映射到现有指标(逐指标验证效度); 后 5 个映射不到 = "缺口探针"(反推我们缺哪个指标)。
FAULT_CODES = frozenset((
    "contact_late", "contact_early", "contact_low", "contact_high",   # → contact_forward/height
    "arm_only", "no_load", "rotate_late",                             # → seq_lead/xfactor/rot_pre
    "footwork", "finish", "balance", "tempo", "other",               # → 无指标(缺口探针)
))
MAX_FAULTS = 6                    # 一拍最多记几个芯片(防手滑全选)

# 容器内路径; 部署时用 docker run -v /home/ubuntu/coach_data:/app/coach_data 挂载宿主目录
COACH_DIR = os.environ.get("COACH_DATA_DIR",
                           os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "coach_data"))


def _load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _codes() -> dict:
    return _load_json(os.path.join(COACH_DIR, "codes.json"), {})


def _manifest() -> list:
    return _load_json(os.path.join(COACH_DIR, "manifest.json"), [])


def _rating_path(code: str) -> str:
    return os.path.join(COACH_DIR, "ratings", f"{code}.json")


def _load_rating(code: str) -> dict:
    return _load_json(_rating_path(code),
                      {"code": code, "started_at": None, "ratings": {}, "submitted": False})


def _save_rating(code: str, doc: dict) -> None:
    os.makedirs(os.path.join(COACH_DIR, "ratings"), exist_ok=True)
    p = _rating_path(code)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def _auth(code: str) -> dict:
    info = _codes().get(code or "")
    if not info:
        raise HTTPException(403, "授权码无效")
    return info


def _clip_order(code: str, items: list) -> list:
    """同一批片段, 每位教练确定性洗牌(seed=码)——断点续评顺序稳定, 教练间顺序不同。"""
    order = list(items)
    random.Random(code).shuffle(order)
    return order


class LoginReq(BaseModel):
    code: str


class RateReq(BaseModel):
    code: str
    clip_id: str
    score: int           # 1~10 总体
    good: bool           # 好/差 二选一
    faults: list[str] = []   # 原因芯片 code(有序, 第一个=主要问题); 落盘按白名单过滤
    clarity: int = 0     # 拍摄清晰度 0未答/1看不清/2一般/3清楚(剔除拍摄质量混淆)
    comment: str = ""
    seconds: float = 0.0  # 该条停留时长(防敷衍核查用)


class SubmitReq(BaseModel):
    code: str


@router.post("/login")
def login(req: LoginReq):
    info = _auth(req.code)
    items = _manifest()
    if not items:
        return {"ok": False, "reason": "评测集尚未就绪, 请稍后再来"}
    doc = _load_rating(req.code)
    if doc.get("started_at") is None:
        doc["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        doc["name"] = info.get("name", "")
        _save_rating(req.code, doc)
    order = _clip_order(req.code, [it["id"] for it in items])
    return {"ok": True, "name": info.get("name", ""),
            "clips": order,                        # 只有 id 顺序, 无任何隐藏字段
            "done": sorted(doc["ratings"].keys()),
            "submitted": bool(doc.get("submitted"))}


@router.get("/clip/{clip_id}")
def clip(clip_id: str, code: str):
    _auth(code)
    it = next((x for x in _manifest() if x["id"] == clip_id), None)
    if it is None:
        raise HTTPException(404, "无此片段")
    path = os.path.normpath(os.path.join(COACH_DIR, it["file"]))
    if not path.startswith(os.path.normpath(COACH_DIR)) or not os.path.isfile(path):
        raise HTTPException(404, "片段文件缺失")
    return FileResponse(path, media_type="video/mp4")


@router.post("/rate")
def rate(req: RateReq):
    _auth(req.code)
    if not any(x["id"] == req.clip_id for x in _manifest()):
        raise HTTPException(404, "无此片段")
    if isinstance(req.score, bool) or not 1 <= req.score <= 10:
        raise HTTPException(422, "score 需在 1~10")
    with _LOCK:
        doc = _load_rating(req.code)
        if doc.get("submitted"):
            raise HTTPException(409, "已最终提交, 不可修改")
        # 芯片按白名单过滤+去重保序(第一个=主要问题), 截断到 MAX_FAULTS; 未知 code 丢弃
        seen = set()
        faults = [f for f in (req.faults or [])
                  if f in FAULT_CODES and not (f in seen or seen.add(f))][:MAX_FAULTS]
        clarity = int(req.clarity) if req.clarity in (0, 1, 2, 3) else 0
        doc["ratings"][req.clip_id] = {
            "score": int(req.score), "good": bool(req.good),
            "faults": faults, "clarity": clarity,
            "comment": (req.comment or "")[:500],
            "seconds": round(float(req.seconds), 1),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_rating(req.code, doc)
    return {"ok": True, "done": len(doc["ratings"]), "total": len(_manifest())}


@router.post("/submit")
def submit(req: SubmitReq):
    _auth(req.code)
    with _LOCK:
        doc = _load_rating(req.code)
        missing = [it["id"] for it in _manifest() if it["id"] not in doc["ratings"]]
        if missing:
            raise HTTPException(422, f"还有 {len(missing)} 条未评")
        doc["submitted"] = True
        doc["submitted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _save_rating(req.code, doc)
    return {"ok": True}
