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
    "seq_lead": {
        "label": "发力链顺序 (近端→远端)", "unit": "", "direction": "band",
        "good": "下盘/躯干先发力, 再依次传到手臂, 动力链顺畅、与德约接近。",
        "below": "近端领先不足(几乎同时发力) → 偏'用手臂打'。引拍后先转髋顶髋, 让手臂稍晚依次跟随甩出。",
        "above": "近端领先偏多, 注意各环节衔接别脱节。",
    },
    "xfactor_magnitude": {
        "label": "X-factor 装载幅度", "unit": "°", "direction": "band",
        "good": "上下半身分离充分, 蓄力到位。",
        "above": "肩髋分离偏大, 注意别过度扭转导致还原慢或腰部负担。",
        "below": "上下半身分离不足, 蓄力偏小。引拍时多转肩、稳住下盘, 制造更大的肩髋夹角来储能。",
    },
    "contact_forward": {
        "label": "击球点·前伸", "unit": "", "direction": "band",
        "good": "击球点在身体前方的位置和德约接近, 能充分借上身体前送的力。",
        "below": "击球点偏靠后(离身体太近) → 容易被球顶住、发不上力。让击球点更靠前, 早一点迎击。",
        "above": "击球点过于靠前, 可能够不实、发力不稳, 注意触球时机。",
    },
    "contact_height": {
        "label": "击球点·高度", "unit": "", "direction": "band",
        "good": "击球高度和德约接近, 处在舒适发力区间。",
        "below": "击球点偏低 → 多在腰部以下, 注意降重心、早准备, 或选更高的击球点。",
        "above": "击球点偏高, 注意是否被高球顶到、影响上旋与稳定。",
    },
}
ORDER = ["seq_lead", "xfactor_magnitude", "contact_forward", "contact_height"]
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
    rpt = (reference.get("reference", {}) or {}).get("peak_times") or {}
    sequencing = [{"seg": s, "label": SEQ_LABELS[s],
                   "t": pt.get(s), "ref_t": rpt.get(s)}
                  for s in SEQ_ORDER if pt.get(s) is not None]

    return {"score": score, "summary": summary, "items": items,
            "sequencing": sequencing}
