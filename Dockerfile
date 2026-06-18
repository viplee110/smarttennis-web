FROM python:3.11-slim

# 可选 apt 镜像: 国内构建传 --build-arg APT_MIRROR=mirrors.tencentyun.com 加速 (默认走官方源)
ARG APT_MIRROR=
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|deb.debian.org|$APT_MIRROR|g; s|security.debian.org|$APT_MIRROR|g" \
            /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
        sed -i "s|deb.debian.org|$APT_MIRROR|g; s|security.debian.org|$APT_MIRROR|g" \
            /etc/apt/sources.list 2>/dev/null || true; \
    fi

# MediaPipe / OpenCV / 视频解码所需的系统库
# 注意: MediaPipe 原生库依赖 GL/GLES (libGLESv2.so.2, libEGL.so.1)，缺则 create_from_options 报 OSError
# fonts-wqy-zenhei: 中文字体(~16MB), 否则 matplotlib 图表里中文(髋/肩/你/德约…)全变成豆腐块/英文
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libglib2.0-0 libgl1 libgles2 libegl1 libopengl0 \
        fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖以利用层缓存
# 可选 PyPI 镜像: 国内构建传 --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple 加速
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -i ${PIP_INDEX_URL} -r backend/requirements.txt

# 预热 matplotlib 字体缓存, 并在构建日志里列出 matplotlib 实际识别到的中文字体名。
# 检测逻辑与运行期 shadow._use_cjk_font() 的标记保持一致; 找不到则打印 CJK_FONT_MISSING(图表会退回英文)。
RUN python -c "from matplotlib import font_manager as fm; M=('yahei','simhei','simsun','song','noto sans cjk','wenquanyi','wqy','zenhei','zen hei'); cn=sorted({f.name for f in fm.fontManager.ttflist if any(m in f.name.lower() for m in M)}); print('matplotlib CJK fonts:', cn); print('CJK_FONT_OK' if cn else 'CJK_FONT_MISSING')"

COPY . .

# Hugging Face Spaces 默认 7860; Render/Railway 会注入 PORT
ENV PORT=7860 \
    MPLBACKEND=Agg
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --app-dir backend --host 0.0.0.0 --port ${PORT:-7860}"]
