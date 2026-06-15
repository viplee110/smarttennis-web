"""
build_reference.py — 从多条德约正手挥拍预计算诊断基准 (IQR band)
================================================================
读取 landmarks/ 下的逐条 stroke JSON, 对每条跑 kinetic_chain.analyze,
汇总三项指标的四分位区间 (绿带), 并保存 stroke_027 的 contact 姿态与
理想动力链曲线, 作为前端"影子对比"与图表参考。

用法:
    python build_reference.py --src "<Build>/landmarks" --out reference/djokovic_forehand.json
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import kinetic_chain as kc

METRIC_KEYS = ["hip_to_forearm_lag", "xfactor_magnitude", "xfactor_release"]
# 已人工验证的基准条 (用完整文件名精确匹配, 避免其他视频的同名 stroke_027 误覆盖)
REF_STROKE = "Novak_Djokovic_Forehand_Slow_Motion__stroke_027"
# 人工核对的真·触球帧: 自动检测给 130 (尚在拍头下降), 137 才是拍面触球。
# 仅用于参考姿态/理想曲线的锚点; 绿带统计仍用各条自动 contact 保持一致。
REF_CONTACT = 137


def _iqr(vals: list[float]) -> dict:
    a = np.array(vals, dtype=float)
    q25, q50, q75 = (float(np.percentile(a, p)) for p in (25, 50, 75))
    return {"q25": q25, "median": q50, "q75": q75, "lo": q25, "hi": q75,
            "min": float(a.min()), "max": float(a.max()), "n": len(vals)}


def build(src_dir: str, out_path: str) -> dict:
    files = sorted(glob.glob(os.path.join(src_dir, "*__stroke_*.json")))
    print(f"找到 {len(files)} 条 stroke")

    samples: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    ref_payload = None
    used = 0
    for fp in files:
        try:
            data = json.load(open(fp, encoding="utf-8"))
            res = kc.analyze(data)
        except Exception as e:                       # noqa: BLE001
            print(f"  [skip] {os.path.basename(fp)}: {e}")
            continue
        if res["valid_ratio"] < 0.8 or len(data["frames"]) < 30:
            continue                                  # 检测质量差/太短的丢弃
        m = res["metrics"]
        for k in METRIC_KEYS:
            samples[k].append(m[k])
        used += 1

        if os.path.splitext(os.path.basename(fp))[0] == REF_STROKE:
            res_ref = kc.analyze(data, contact_override=REF_CONTACT)   # 锚定真·触球帧
            ref_payload = _extract_reference_pose(data, res_ref)

    if ref_payload is None and files:                 # 兜底: 用第一条做参考姿态
        d0 = json.load(open(files[0], encoding="utf-8"))
        ref_payload = _extract_reference_pose(d0, kc.analyze(d0))

    band = {k: _iqr(samples[k]) for k in METRIC_KEYS}
    out = {
        "player": "Novak Djokovic", "stroke": "forehand",
        "n_strokes": used, "metrics_band": band,
        "reference": ref_payload,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\n用了 {used} 条 → {out_path}")
    for k in METRIC_KEYS:
        b = band[k]
        print(f"  {k:22s} band[{b['lo']:.2f}, {b['hi']:.2f}] median={b['median']:.2f}")
    return out


def _extract_reference_pose(data: dict, res: dict) -> dict:
    """提取参考条在 contact 帧的 2D 骨架 (img 归一化坐标) 与理想动力链曲线。"""
    frames = data["frames"]
    cf = res["contact"]
    img = frames[cf]["img"] if frames[cf].get("img") else None
    s = res["signals"]
    # 下采样曲线以减小体积; 时间轴改为"相对击球瞬间"(contact=0), 便于与用户叠加
    import numpy as np
    t_rel = np.asarray(s["t"]) - float(s["t"][res["contact_local"]])
    step = max(1, len(t_rel) // 60)
    curve = {
        "t": [round(float(x), 3) for x in t_rel[::step]],
        "loading_s": round(float(res.get("loading_s", 0.0)), 4),   # 装载时长, 供相位归一化
        "xfactor": [round(float(x), 2) for x in s["xfactor"][::step]],
        **{k: [round(float(x), 3) for x in s["norm"][k][::step]]
           for k in ["hip", "shoulder", "upper_arm", "forearm", "wrist"]},
    }
    return {
        "stroke": REF_STROKE, "hand": s["hand"],
        "contact_frame": int(cf), "contact_local": int(res["contact_local"]),
        "contact_pose_img": [[round(p[0], 5), round(p[1], 5)] for p in img] if img else None,
        "ideal_curve": curve,
        "metrics": {k: round(float(res["metrics"][k]), 4) for k in METRIC_KEYS},
    }


if __name__ == "__main__":
    default_src = r"C:\Dropbox\AI projects\SmartTennis\SmartTennis Build\landmarks"
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=default_src)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "reference", "djokovic_forehand.json"))
    args = ap.parse_args()
    build(args.src, args.out)
