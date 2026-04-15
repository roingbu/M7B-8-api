#!/usr/bin/env bash
# =============================================================================
# AOI 多通道 YOLO：先 DOT 检测（4 卡 DDP），成功后再 MURA 分割（4 卡 DDP）
#
# 显存 16G×4、输入约 (9|8)×640×640 时，全局 batch 宜小；
#
# 使用 screen（避免 SSH 断开丢训练）：
#   screen -S aoi_yolo
#   ./train_aoi_screen.sh
#   # 断开后重连: screen -r aoi_yolo
#
# 或直接（会尝试安装 screen，并提示若未在 screen 内）：
#   ./train_aoi_screen.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- 可选：安装 screen ---
if ! command -v screen &>/dev/null; then
  if command -v apt-get &>/dev/null; then
    echo "[train_aoi_screen] 正在安装 screen …"
    sudo apt-get update -qq
    sudo apt-get install -y screen
  else
    echo "[train_aoi_screen] 未找到 screen 且无 apt-get，请自行安装 screen 后再运行。"
    exit 1
  fi
fi

# --- 提醒使用 screen ---
if [[ -z "${STY:-}" ]]; then
  echo "=================================================================="
  echo "  建议先进入 screen 再训练，避免终端断开导致进程被挂起/丢失："
  echo "    screen -S aoi_yolo"
  echo "    cd $ROOT && ./train_aoi_screen.sh"
  echo "=================================================================="
  sleep 2
fi

# --- 可调环境变量（每项均可 export 覆盖；未设置则用 :- 后的默认值）---

# 是否在多卡 DDP 下把普通 BN 换成 SyncBatchNorm（各卡统计量同步）；0=关闭，1=开启
export YOLO_SYNC_BN="${YOLO_SYNC_BN:-1}" # 1=启用跨卡 SyncBatchNorm

# DOT 检测：预训练权重路径（默认 yolo26m.pt；若仅写模型名 Ultralytics 可自动下载）
export MODEL_DOT="${MODEL_DOT:-/roing-workspace/ultralytics-aoi/yolo26m.pt}" # 可设置 /path/to/yolo26m.pt
# DOT：训练总轮数
export DOT_EPOCHS="${DOT_EPOCHS:-200}" # 训练轮次
# DOT：全局 batch = 所有 GPU 本步样本数之和（4 卡时≈每卡 batch×4；显存不够就减小）
export DOT_BATCH="${DOT_BATCH:-8}" # 全局 batch size
# DOT：名义 batch（nbs）；Ultralytics 用 round(nbs/全局batch) 决定梯度累加步数，并缩放 loss/weight_decay
export DOT_NBS="${DOT_NBS:-64}" # 用于自动计算梯度累加
# DOT：本次实验在 runs/dot_detect 下的子目录名（与 project 组合成输出路径）
export DOT_NAME="${DOT_NAME:-mc9_4gpu_try0}" # 输出实验名
# DOT：是否从上一次训练的 last.pt 继续
export DOT_RESUME="${DOT_RESUME:-0}" # 1=从 last.pt 继续训练

# MURA 分割：预训练权重（可为文件名如 yolo26n-seg.pt，不存在时 Ultralytics 会尝试下载）
export MODEL_SEG="${MODEL_SEG:-/roing-workspace/ultralytics-aoi/yolo26m-seg.pt}"
# MURA：训练总轮数
export SEG_EPOCHS="${SEG_EPOCHS:-200}" # 训练轮次
# MURA：全局 batch（分割+mask 更占显存，通常比 DOT 更小）
export SEG_BATCH="${SEG_BATCH:-4}" # 全局 batch size
# MURA：名义 batch，作用同 DOT_NBS（与 SEG_BATCH 搭配控制梯度累加）
export SEG_NBS="${SEG_NBS:-32}" # 用于自动计算梯度累加
# MURA：本次实验在 runs/mura_segment 下的子目录名
export SEG_NAME="${SEG_NAME:-mc8_4gpu_try0}" # 输出实验名
# MURA：是否从上一次训练的 last.pt 继续
export SEG_RESUME="${SEG_RESUME:-0}" # 1=从 last.pt 继续训练

# 透传给 train_dot_detect.py / train_mura_segment.py 的额外 CLI，例如 "--no-amp" 关闭混合精度
EXTRA_TRAIN_FLAGS="${EXTRA_TRAIN_FLAGS:-}"

echo "========== 1/2 DOT 检测 (4 GPU) =========="
if [[ "$MODEL_DOT" == */* ]] && [[ ! -f "$MODEL_DOT" ]]; then
  echo "错误: 未找到检测权重: $MODEL_DOT"
  echo "请放置 yolo26m.pt 到指定路径，或改用 export MODEL_DOT=yolo26m.pt 让 Ultralytics 自动下载。"
  exit 1
fi

echo "========== 2/2 MURA 检测 (4 GPU) =========="
if [[ "$MODEL_SEG" == */* ]] && [[ ! -f "$MODEL_SEG" ]]; then
  echo "错误: 未找到分割权重: $MODEL_SEG"
  echo "请放置 yolo26m-seg.pt 到指定路径，或改用 export MODEL_SEG=yolo26m-seg.pt 让 Ultralytics 自动下载。"
  exit 1
fi

python3 train_dot_detect.py \
  --model "$MODEL_DOT" \
  --device 0,1,2,3 \
  --batch "$DOT_BATCH" \
  --nbs "$DOT_NBS" \
  --epochs "$DOT_EPOCHS" \
  --name "$DOT_NAME" \
  --workers 8 \
  $( [[ "$DOT_RESUME" == "1" ]] && echo "--resume" ) \
  $EXTRA_TRAIN_FLAGS

echo "========== 2/2 MURA 分割 (4 GPU) =========="
python3 train_mura_segment.py \
  --model "$MODEL_SEG" \
  --device 0,1,2,3 \
  --batch "$SEG_BATCH" \
  --nbs "$SEG_NBS" \
  --epochs "$SEG_EPOCHS" \
  --name "$SEG_NAME" \
  --workers 8 \
  $( [[ "$SEG_RESUME" == "1" ]] && echo "--resume" ) \
  $EXTRA_TRAIN_FLAGS

echo "========== 全部完成 =========="
