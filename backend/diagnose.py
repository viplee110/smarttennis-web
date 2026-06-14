"""
diagnose.py — 把用户指标对照德约 IQR band 生成诊断报告
=====================================================
每项指标判定 below / in_band / above, 配中文教练建议与总体评分。
"""
from __future__ import annotations

# 每项指标的语义与中文文案。direction: "lower"=越低越好(向德约靠拢),
# "band"=落在区间内最佳。
METRIC_META = {
    "hip_to_forearm_lag": {
        "label": "髋-前臂时序 (紧凑度)", "unit": "", "direction": "lower",
        "good": "发力链紧凑, 髋与手臂几乎同步释放, 和德约一致。",
        "above": "手臂相对躯干抢跑/拖沓, 发力链脱节。试着用躯干带动手臂, 引拍后让髋先转、手臂跟随, 而不是单独抡手臂。",
        "below": "时序非常紧凑。",
    },
    "xfactor_magnitude": {
        "label": "X-factor 装载幅度", "unit": "°", "direction": "band",
        "good": "上下半身分离充分, 蓄力到位。",
        "above": "肩髋分离偏大, 注意别过度扭转导致还原慢或腰部负担。",
        "below": "上下半身分离不足, 蓄力偏小。引拍时多转肩、稳住下盘, 制造更大的肩髋夹角来储能。",
    },
    "xfactor_release": {
        "label": "X-factor 释放 (击球时机)", "unit": "°", "direction": "band",
        "good": "击球瞬间分离释放时机合适, 力量顺畅传导到球。",
        "above": "击球时分离释放不足/过早, 容易只用手臂打。让髋带动躯干充分回正再触球。",
        "below": "击球时过度反向, 时机偏晚。",
    },
}

ORDER = ["hip_to_forearm_lag", "xfactor_magnitude", "xfactor_release"]


def _verdict(value: float, band: dict, direction: str) -> tuple[str, bool]:
    """返回 (状态, 是否OK)。"""
    lo, hi = band["lo"], band["hi"]
    if value < lo:
        status = "below"
    elif value > hi:
        status = "above"
    else:
        return "in_band", True
    if direction == "lower":
        # 越低越好: 低于区间也算优秀
        return ("in_band", True) if status == "below" else (status, False)
    return status, False


def diagnose(user_metrics: dict, reference: dict) -> dict:
    band_all = reference["metrics_band"]
    items, n_ok = [], 0
    for key in ORDER:
        meta = METRIC_META[key]
        band = band_all[key]
        val = float(user_metrics[key])
        status, ok = _verdict(val, band, meta["direction"])
        n_ok += int(ok)
        if status == "in_band":
            tip = meta["good"]
        elif status == "above":
            tip = meta["above"]
        else:
            tip = meta["below"]
        items.append({
            "key": key, "label": meta["label"], "unit": meta["unit"],
            "value": round(val, 2), "band_lo": round(band["lo"], 2),
            "band_hi": round(band["hi"], 2), "median": round(band["median"], 2),
            "status": status, "ok": ok, "tip": tip,
        })
    score = round(100 * n_ok / len(ORDER))
    if score >= 80:
        summary = "动力链整体接近职业水准, 继续保持。"
    elif score >= 50:
        summary = "动力链基础不错, 有 1–2 个环节可重点打磨。"
    else:
        summary = "发力链存在明显改进空间, 建议从下半身带动开始练。"
    return {"score": score, "n_ok": n_ok, "n_total": len(ORDER),
            "summary": summary, "items": items}
