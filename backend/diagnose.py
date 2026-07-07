"""
diagnose.py — 把用户指标对照德约 IQR band 生成诊断报告
=====================================================
连续评分(非二值): 每项按"与德约中位数的距离/区间宽度"算 0–100 子分,
总分=各项均值。并把"发力顺序"(各环节峰值时刻)摊开展示, 去黑箱。
"""
from __future__ import annotations

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
    "rot_pre_frac": {
        "label": "击球前身体旋转完成度", "unit": "", "direction": "higher",
        "good": "击球前髋肩已转到位, 触球时把旋转能量交给手臂(德约式)。",
        "below": "身体旋转偏晚——大部分髋肩旋转发生在击球之后(转过了球)。"
                 "练: 引拍后先转髋再转肩、在触球前完成转体, 触球时身体已正对来球方向、随即把力甩给手臂。",
        "above": "",
    },
}
ORDER = ["seq_lead", "xfactor_magnitude", "contact_forward", "contact_height", "rot_pre_frac"]
SEQ_LABELS = {"hip": "髋", "shoulder": "肩", "upper_arm": "上臂",
              "forearm": "前臂", "wrist": "手腕"}
SEQ_ORDER = ["hip", "shoulder", "upper_arm", "forearm", "wrist"]


# 时序类指标可靠所需的最少装载帧数(引拍底→击球)。30fps 常速业余挥拍通常只有 3~6 帧,
# 量化噪声≈带宽级(正控实验: 同一挥拍两次测 seq_lead 差 0.2+), 低于此阈值锁定不评。
TIMING_MIN_LOADING = 10
# rot_pre 也锁: 短装载下窗口相位失真, 连德约本人都只测出 0.14(正控回归实测)——
# 它反映的是拍摄条件而非技术, 30fps 常速下如实只评击球点两项。
TIMING_LOCKED = ("seq_lead", "xfactor_magnitude", "rot_pre_frac")
LOCK_TIP = ("此指标需要足够的时间分辨率——你的前挥只有几帧, 精细时序物理上测不出来。"
            "用手机慢动作模式(120/240fps)拍摄即可解锁。")


def _acceptable(band: dict):
    """合格区 [P10,P90]。旧版参考带无 p10/p90 时按近正态从四分位外推(×1.9)。"""
    q1, q3, med = band["lo"], band["hi"], band["median"]
    p10 = band.get("p10", med - 1.9 * max(med - q1, 1e-6))
    p90 = band.get("p90", med + 1.9 * max(q3 - med, 1e-6))
    return p10, p90


def _subscore(val: float, band: dict, direction: str) -> float:
    """0–100 连续子分。IQR(中间50%)内=100; IQR边→P10/P90 线性 100→75;
    合格区外再一个半合格区宽度线性衰减到 0。
    设计依据: IQR 当合格线会把一半职业挥拍判"出带"(正控实验), 合格区必须是 P10-P90。"""
    q1, q3, med = band["lo"], band["hi"], band["median"]
    if direction == "higher" and val >= med:
        return 100.0
    if direction == "lower" and val <= med:
        return 100.0
    if q1 <= val <= q3:
        return 100.0
    p10, p90 = _acceptable(band)
    edge, outer = (q1, p10) if val < q1 else (q3, p90)
    span = max(abs(outer - edge), 1e-6)
    d = abs(val - edge)
    if d <= span:
        return round(100.0 - 25.0 * d / span, 1)
    tail = max((p90 - p10) / 2.0, 1e-6)
    return round(max(0.0, 75.0 - 75.0 * (d - span) / tail), 1)


def _status(val: float, band: dict) -> str:
    if band["lo"] <= val <= band["hi"]:
        return "in_band"
    p10, p90 = _acceptable(band)
    if p10 <= val <= p90:
        return "acceptable"
    return "below" if val < band["lo"] else "above"


def diagnose(user_metrics: dict, reference: dict,
             loading_frames: int = None, n_swings: int = 1) -> dict:
    """loading_frames: 装载帧数(时序指标置信门, None=不启用锁);
    n_swings: 聚合的挥拍数(>=2 时说明评分基于多挥拍中位数)。"""
    band_all = reference["metrics_band"]
    timing_locked = loading_frames is not None and loading_frames < TIMING_MIN_LOADING
    items, subs, weights = [], [], []
    locked_keys = []
    for key in ORDER:
        meta = METRIC_META[key]
        band = band_all[key]
        val = float(user_metrics[key])
        locked = timing_locked and key in TIMING_LOCKED
        p10, p90 = _acceptable(band)
        if locked:
            locked_keys.append(key)
            items.append({
                "key": key, "label": meta["label"], "unit": meta["unit"],
                "value": round(val, 2), "band_lo": round(band["lo"], 2),
                "band_hi": round(band["hi"], 2), "median": round(band["median"], 2),
                "p10": round(p10, 2), "p90": round(p90, 2),
                "score": None, "status": "low_confidence", "ok": None,
                "locked": True, "tip": LOCK_TIP,
            })
            continue
        sub = _subscore(val, band, meta["direction"])
        subs.append(sub)
        weights.append(1.0)
        status = _status(val, band)
        better_side = (meta["direction"] == "higher" and val > band["hi"]) or \
                      (meta["direction"] == "lower" and val < band["lo"])
        if status == "in_band" or better_side:
            tip, ok = meta["good"], True
        elif status == "acceptable":
            tip, ok = meta["good"] + "(在职业常见波动范围内, 离中位还有些距离)", True
        elif status == "above":
            tip, ok = meta["above"], False
        else:
            tip, ok = meta["below"], False
        items.append({
            "key": key, "label": meta["label"], "unit": meta["unit"],
            "value": round(val, 2), "band_lo": round(band["lo"], 2),
            "band_hi": round(band["hi"], 2), "median": round(band["median"], 2),
            "p10": round(p10, 2), "p90": round(p90, 2),
            "score": sub, "status": status, "ok": ok,
            "locked": False, "tip": tip,
        })

    score = round(sum(subs) / max(sum(weights), 1e-6))
    if score >= 80:
        summary = "动力链整体接近职业水准, 继续保持。"
    elif score >= 60:
        summary = "动力链基础不错, 有 1–2 个环节可重点打磨。"
    else:
        summary = "发力链存在明显改进空间, 建议从下半身带动开始练。"
    prefix = f"基于 {n_swings} 次挥拍综合评估。" if n_swings >= 2 else ""
    suffix = (f"(本次可评 {len(subs)}/{len(ORDER)} 项, 时序类需慢动作视频解锁)"
              if locked_keys else "")
    summary = prefix + summary + suffix

    # 发力顺序透明展示 (各环节峰值相对击球的时刻)
    pt = user_metrics.get("peak_times") or {}
    rpt = (reference.get("reference", {}) or {}).get("peak_times") or {}
    sequencing = [{"seg": s, "label": SEQ_LABELS[s],
                   "t": pt.get(s), "ref_t": rpt.get(s)}
                  for s in SEQ_ORDER if pt.get(s) is not None]

    return {"score": score, "summary": summary, "items": items,
            "sequencing": sequencing, "n_swings": n_swings,
            "locked": locked_keys}
