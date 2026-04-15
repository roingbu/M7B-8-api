#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="aoi"

conda activate "$ENV_NAME"

# 安装 PyTorch cu126（支持 sm_60）
pip install --upgrade pip
pip uninstall -y xformers flash-attn || true
pip install numpy<2
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126

pip install ultralytics