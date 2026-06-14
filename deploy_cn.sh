#!/usr/bin/env bash
# 国内服务器一键部署 (腾讯云轻量 / 阿里云等)。在仓库根目录运行: sudo bash deploy_cn.sh
set -e

echo "==> 1/3 配置 Docker 镜像加速 (腾讯云内网源)"
sudo mkdir -p /etc/docker
echo '{"registry-mirrors":["https://mirror.ccs.tencentyun.com","https://docker.m.daocloud.io"]}' \
  | sudo tee /etc/docker/daemon.json >/dev/null
sudo systemctl restart docker

echo "==> 2/3 构建镜像 (pip 走清华源, 约 3-6 分钟)"
sudo docker build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t smarttennis .

echo "==> 3/3 启动容器 (宿主 80 端口 -> 容器 7860)"
sudo docker rm -f smarttennis 2>/dev/null || true
sudo docker run -d --restart unless-stopped -p 80:7860 --name smarttennis smarttennis

echo ""
echo "✅ 部署完成。手机浏览器打开:  http://<你的公网IP>"
echo "   查看日志:  sudo docker logs -f smarttennis"
