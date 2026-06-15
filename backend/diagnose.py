"""
diagnose.py — 把用户指标对照德约 IQR band 生成诊断报告
=====================================================
连续评分(非二值): 每项按"与德约中位数的距离/区间宽度"算 0–100 子分,
总分=各项均值。并把"发力顺序"(各环节峰值时刻)摊开展示, 去黑箱。
"""
from __future__ import annotations
import math

# direction: "higher"=越大越好(到中位即满分), "band"=越接近德约区间越好。
METRIC_META = {
    "hip_to_forearm_lag": {
        "label": "发力时序 (髋领先手臂)", "unit": "", "direction": "higher",
        "good": "髋明显领先手臂发力, 力从下盘传到手臂, 动力链顺畅。",
        "below": "髋和手臂几乎同时发力 → 在'用手臂打球'。试着引拍后先转髋顶髋, 让手臂稍晚跟随甩出。",
        "above": "时序非常充分。",
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
SEQ_LABELS = {"hip": "髋", "shoulder": "肩", "upper_arm": "上臂",
              "forearm": "前臂", "wrist": "手腕"}
SEQ_ORDER = ["hip", "shoulder", "upper_arm", "forearm", "wrist"]


def _subscore(val: float, band: dict, direction: str) -> float:
    """0–100 连续子分: 越接近德约中位越高 (按区间半宽做高斯衰减)。
    higher/lower 方向上, 处在'更好'一侧直接满分。"""
    lo, hi, med = band["lo"], band["hi"], band["median"]
    half = max((hi - lo) / 2.0, 1e-6)
    if direction == "higher" and val >= med:
        return 100.0
    if direction == "lower" and val <= med:
        return 100.0
    z = (val - med) / half
    return round(100.0 * math.exp(-0.5 * z * z), 1)


def _status(val: float, band: dict, direction: str) -> str:
    if band["lo"] <= val <= band["hi"]:
        return "in_band"
    return "below" if val < band["lo"] else "above"


def diagnose(user_metrics: dict, reference: dict) -> dict:
    band_all = reference["metrics_band"]
    items, subs = [], []
    for key in ORDER:
        meta = METRIC_META[key]
        band = band_all[key]
        val = float(user_metrics[key])
        sub = _subscore(val, band, meta["direction"])
        subs.append(sub)
        status = _status(val, band, meta["direction"])
        # "更好一侧" 视为正常
        if status == "in_band" or (meta["direction"] == "higher" and val > band["hi"]) \
                or (meta["direction"] == "lower" and val < band["lo"]):
            tip, ok = meta["good"], True
        elif status == "above":
            tip, ok = meta["above"], sub >= 75
        else:
            tip, ok = meta["below"], sub >= 75
        items.append({
            "key": key, "label": meta["label"], "unit": meta["unit"],
            "value": round(val, 2), "band_lo": round(band["lo"], 2),
            "band_hi": round(band["hi"], 2), "median": round(band["median"], 2),
            "score": sub, "status": status, "ok": ok, "tip": tip,
        })

    score = round(sum(subs) / len(subs))
    if score >= 80:
        summary = "动力链整体接近职业水准, 继续保持。"
    elif score >= 60:
        summary = "动力链基础不错, 有 1–2 个环节可重点打磨。"
    else:
        summary = "发力链存在明显改进空间, 建议从下半身带动开始练。"

    # 发力顺序透明展示 (各环节峰值相对击球的时刻)
    pt = user_metrics.get("peak_times") or {}
    sequencing = [{"seg": s, "label": SEQ_LABELS[s], "t": pt.get(s)}
                  for s in SEQ_ORDER if pt.get(s) is not None]
    sequence_ok = bool(user_metrics.get("sequence_ok", False))

    return {"score": score, "summary": summary, "items": items,
            "sequencing": sequencing, "sequence_ok": sequence_ok}
