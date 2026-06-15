# 🎾 SmartTennis MVP

上传一段 ~10 秒的网球**正手**视频 → 2D 骨骼提取 → 动力链分析 → 对照德约科维奇的诊断报告 + 影子骨架叠加对比。

手机浏览器打开即可现场拍摄并分析，适合面向用户的 demo 展示。

![MediaPipe](https://img.shields.io/badge/MediaPipe-pose-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-backend-green) ![License](https://img.shields.io/badge/license-AGPL--3.0-orange)

---

## 它做什么

1. **骨骼提取** — MediaPipe Pose 逐帧提取 33 个关节点（图像坐标 + 世界坐标）。
2. **动力链分析** — 计算髋转动、肩转动、上臂、前臂、手腕速度的归一化角速度时序，以及 X-factor（肩髋分离角），自动检测击球瞬间（contact）。
3. **诊断报告** — 把用户三项指标对照 **87 条德约正手**预计算的 IQR 正常区间（绿带）：
   - 髋-前臂时序（发力链紧凑度）
   - X-factor 装载幅度
   - X-factor 释放（击球时机）
4. **影子对比** — 把用户与德约在击球瞬间的骨架按身材归一化后叠加。

输出：动力链时序图（你=实线 / 德约=虚线）、影子叠加图、逐项诊断卡片 + 0–100 评分。

---

## 项目结构

```
smarttennis-web/
├── backend/
│   ├── app.py              FastAPI: POST /api/analyze, GET /
│   ├── pose.py             MediaPipe 视频→landmarks
│   ├── kinetic_chain.py    动力链信号 + contact 检测 + 指标
│   ├── diagnose.py         对照 IQR band 出诊断文案
│   ├── shadow.py           服务端渲染动力链图 + 影子叠加 (PNG)
│   ├── build_reference.py  预计算德约 IQR band (开发用)
│   ├── reference/
│   │   └── djokovic_forehand.json   ← 预计算基准 (随仓库分发)
│   ├── models/
│   │   └── pose_landmarker_lite.task
│   └── requirements.txt
├── frontend/
│   └── index.html          手机友好单页 (无构建步骤)
├── Dockerfile              部署无关, 一键构建
└── README.md
```

---

## 本地运行

需要 Python ≥ 3.10。

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate   |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

浏览器打开 http://127.0.0.1:8000 。手机同局域网时用电脑内网 IP（如 `http://192.168.x.x:8000`）即可现场拍摄。

> 拍摄建议：**侧面机位、单人入镜、全身可见，把击球瞬间放在视频中段**，10 秒以内最佳。

---

## 部署

后端是标准 FastAPI + Dockerfile，与平台解耦。容器监听 `PORT`（默认 7860）。

### A. Hugging Face Spaces（免费，适合非大陆观众 / 自己预览）

1. 新建 Space → SDK 选 **Docker**。
2. 把本仓库内容推到该 Space 的 git，或在 GitHub 连接后自动同步。
3. Spaces 默认端口 7860，本 Dockerfile 已对齐，构建完成即得公网 URL。

> 免费层 16GB 内存，足够跑 MediaPipe。**注意：`*.hf.space` 在中国大陆访问不稳定**，现场给国内观众演示请用方案 B。

### B. 国内轻量服务器（面向中国大陆演示，最稳）

腾讯云 Lighthouse / 阿里云轻量（约 ¥24–60/月）：

```bash
# 服务器上
git clone <你的仓库>
cd smarttennis-web
docker build -t smarttennis .
docker run -d -p 80:7860 --restart unless-stopped smarttennis
```

用 `http://公网IP` 直接访问。**用 IP 直连可免 ICP 备案**（仅当你要用域名挂 80/443 才需备案）。

> 手机摄像头调用（前端 `capture`）在 `http://IP` 下可用；若改用 `getUserMedia` 实时取流则需 HTTPS。

---

## 说明与边界

- contact 检测假设挥拍动作大致位于片段中段（中心加权），请按拍摄建议录制。
- 诊断基准来自单一来源的德约慢动作素材自动切片，作为 MVP 参考；后续可扩充多机位、多球员、多拍种。
- 仅支持单人、正手；其他拍种与多人场景为后续迭代项。

---

## License 与商用授权

本项目以 **GNU AGPL-3.0** 开源（见 [LICENSE](LICENSE)）。

- ✅ 欢迎学习、研究、自用、二次开发。
- ⚠️ **AGPL 关键条款**：若你将本项目（或其衍生版本）**作为联网服务对外提供**，必须在相同 AGPL-3.0 许可下**公开你的完整源码（含修改）**。
- 💼 **商用 / 闭源授权**：如需在闭源商业产品中使用而不公开源码，请联系作者获取**单独的商业授权**（双授权）。

> 注：本仓库为面向演示的 MVP。完整产品的核心资产（精修的职业动作数据集、训练模型、评分算法）不在此开源范围内。

© 2026 SmartTennis. Licensed under AGPL-3.0.
