#!/usr/bin/env bash
# =============================================================================
# AOI DOT 检测 YOLO：4 卡 DDP
#
# 显存 16G×4、输入约 9×640×640 时，全局 batch 宜小；
#
# 使用 screen（避免 SSH 断开丢训练）：
#   screen -S aoi_dot
#   ./train_dot_screen-p100.sh
#   # 断开后重连: screen -r aoi_dot
#
# 或直接（会尝试安装 screen，并提示若未在 screen 内）：
#   ./train_dot_screen-p100.sh
# =============================================================================
set -euo pipefail

# ==========================================
# 🛡️ 增加：僵尸进程清理拦截器
# ==========================================
cleanup() {
    echo -e "\n[警告] 检测到 Ctrl+C 或脚本退出，正在强制清理所有显卡上的残留进程..."
    # 强杀当前脚本衍生的所有相关 Python 进程
    pkill -9 -f "train_dot_detect-p100.py" || true
    pkill -9 -f "torch.distributed" || true
    echo "[成功] 显存已释放完毕！"
}
# 绑定拦截器：当接收到 SIGINT(Ctrl+C) 或 SIGTERM 时，立刻执行 cleanup
trap cleanup SIGINT SIGTERM EXIT
# ==========================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# # 让脚本在没有 conda init 的情况下仍可激活 aoi 环境
CONDA_ROOT="/roing-workspace/miniconda3"
ENV_NAME="aoi"
# if ! command -v conda &>/dev/null && [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
#   source "$CONDA_ROOT/etc/profile.d/conda.sh"
# fi
# if command -v conda &>/dev/null; then
#   conda activate "$ENV_NAME"
# else
#   echo "错误：未找到 conda 命令，无法激活 aoi 环境。"
#   exit 1
# fi

# # --- 可选：安装 screen ---
# if ! command -v screen &>/dev/null; then
#   if command -v apt-get &>/dev/null; then
#     echo "[train_dot_screen] 正在安装 screen …"
#     if [[ "$EUID" -eq 0 ]]; then
#       apt-get update -qq
#       apt-get install -y screen
#     else
#       sudo apt-get update -qq
#       sudo apt-get install -y screen
#     fi
#   else
#     echo "[train_dot_screen] 未找到 screen 且无 apt-get，请自行安装 screen 后再运行。"
#     exit 1
#   fi
# fi

# --- 提醒使用 screen ---
if [[ -z "${STY:-}" ]]; then
  echo "=================================================================="
  echo "  建议先进入 screen 再训练，避免终端断开导致进程被挂起/丢失："
  echo "    screen -S aoi_dot"
  echo "    cd $ROOT && ./train_dot_screen-p100.sh"
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
export DOT_BATCH="${DOT_BATCH:-32}" # 全局 batch size，P100 CPU 默认 4
# DOT：名义 batch（nbs）；Ultralytics 用 round(nbs/全局batch) 决定梯度累加步数，并缩放 loss/weight_decay
export DOT_NBS="${DOT_NBS:-128}" # 用于自动计算梯度累加
# DOT：本次实验在 runs/dot_detect 下的子目录名（与 project 组合成输出路径）
export DOT_NAME="${DOT_NAME:-mc9_4gpu_try0}" # 输出实验名
# DOT：是否从上一次训练的 last.pt 继续
export DOT_RESUME="${DOT_RESUME:-0}" # 1=从 last.pt 继续训练

# P100 上默认使用 GPU 0,1,2,3；若是多卡 DDP 可设置 P100_DEVICE=0,1,2,3
export P100_DEVICE="${P100_DEVICE:-0,1,2,3}"
# 透传给 train_dot_detect.py 的额外 CLI，例如 "--no-amp" 关闭混合精度
EXTRA_TRAIN_FLAGS="${EXTRA_TRAIN_FLAGS:---no-amp}"  # P100 默认关闭 AMP

echo "========== DOT 检测 (4 GPU) =========="
if [[ "$MODEL_DOT" == */* ]] && [[ ! -f "$MODEL_DOT" ]]; then
  echo "错误: 未找到检测权重: $MODEL_DOT"
  echo "请放置 yolo26m.pt 到指定路径，或改用 export MODEL_DOT=yolo26m.pt 让 Ultralytics 自动下载。"
  exit 1
fi

python train_dot_detect-p100.py \
  --model "$MODEL_DOT" \
  --device "$P100_DEVICE" \
  --batch "$DOT_BATCH" \
  --nbs "$DOT_NBS" \
  --epochs "$DOT_EPOCHS" \
  --name "$DOT_NAME" \
  --workers 8 \
  $( [[ "$DOT_RESUME" == "1" ]] && echo "--resume" ) \
  $EXTRA_TRAIN_FLAGS

echo "========== DOT 检测完成 =========="