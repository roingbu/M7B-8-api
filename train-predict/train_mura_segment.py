"""
MURA 实例分割训练 (Ultralytics YOLO-Seg，多通道 .npy)

数据为 (H,W,8) uint8；dataset.yaml 需含 channels: 8。首层 3→8 通道：预训练权重在输入维 mean 后 broadcast（同 DOT）。

优先使用仓库内 ultralytics-aoi 源码。

用法:
    cd /roing-workspace/train-predict && python train_mura_segment.py
    python train_mura_segment.py --model yolo26n-seg.pt --epochs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AOI = _REPO / "ultralytics-aoi"
if _AOI.is_dir():
    sys.path.insert(0, str(_AOI))

from ultralytics import YOLO

DATASET_YAML = "/roing-workspace/dataset/cleaned_YOLO_MURA/dataset.yaml"
# 本地若无分割权重，可传 yolo26n-seg.pt / yolo11n-seg.pt 由 Ultralytics 自动下载
DEFAULT_MODEL = "yolo26n-seg.pt"
PROJECT_DIR = "/roing-workspace/train-predict/runs/mura_segment"


def parse_args():
    p = argparse.ArgumentParser(description="MURA 多通道分割训练")
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="预训练分割权重 .pt（首层 3→N 通道：RGB 权重通道维 mean 后 broadcast）",
    )
    p.add_argument("--data", type=str, default=DATASET_YAML)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--batch",
        type=int,
        default=4,
        help="全局 batch（分割更吃显存，4×16G 多通道可先试 4；为总 batch，均分各卡）",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0,1,2,3", help="四卡 DDP 示例: 0,1,2,3")
    p.add_argument(
        "--nbs",
        type=int,
        default=16,
        help="名义 batch；如 batch=4、nbs=16 → 梯度累加约 4 步再 optimizer.step",
    )
    p.add_argument("--no-amp", action="store_true", help="关闭 AMP（默认开启）")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name", type=str, default="exp")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--patience", type=int, default=100)
    p.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        choices=["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam"],
    )
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--cos-lr", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.resume:
        last_ckpt = sorted(Path(PROJECT_DIR).rglob("weights/last.pt"))
        if not last_ckpt:
            raise FileNotFoundError(f"未找到 checkpoint: {PROJECT_DIR}")
        model = YOLO(str(last_ckpt[-1]))
        model.train(resume=True)
        return

    if not Path(args.data).is_file():
        raise FileNotFoundError(f"数据配置不存在: {args.data}（请先运行 clean_dataset.py）")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        task="segment",
        epochs=args.epochs,
        batch=args.batch,
        nbs=args.nbs,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        amp=not args.no_amp,
        project=PROJECT_DIR,
        name=args.name,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=args.cos_lr,

        # ==========================================
        # 👑 MURA 进阶专属调参：弱化画框，强化像素级边缘
        # ==========================================
        box=2.0,             # (默认7.5) 调低！MURA边缘极其模糊，强迫模型精准回归矩形框没意义，反而会干扰训练。
        cls=1.0,             # (默认0.5) 稍微调高！确保模型先认对“这是哪种色斑/黑雾”，再去做分割。
        retina_masks=True,   # (默认False) 核心！开启原分辨率高精度 Mask。解决边缘锯齿、多圈/胖一圈的致命武器。
        overlap_mask=False,  # (默认True) 核心！工业面板缺陷不像自然图像有“遮挡合并”，关掉它确保不同缺陷的掩码绝对独立计算。

        # --- 开启安全且符合物理规律的空间变换 ---
        fliplr=0.5,        # 水平翻转（极度安全）
        flipud=0.5,        # 垂直翻转（极度安全）
        translate=0.1,     # 平移（防止缺陷全在中心，安全）
        degrees=0.0,       # 旋转（面板是严格对齐的晶格，如果机台有轻微角度公差可以给 2.0，否则保持 0）
        
        # --- 坚决关闭破坏性特征变换 ---
        hsv_h=0.0,         # 严禁多通道色彩扭曲
        hsv_s=0.0,
        hsv_v=0.0,
        scale=0.0,         # 严禁尺寸缩放（保护绝对像素尺寸）
        mosaic=0.0,        # 关闭马赛克（防止小目标丢失与边缘截断）
        mixup=0.0,         # 关闭像素混合
        copy_paste=0.0,    # 关闭复制粘贴
        
        save=True,
        save_period=10,
        plots=True,
        val=True,
    )

    metrics = model.val()
    print(f"\n{'=' * 50}")
    print(f"Box  mAP50:    {metrics.box.map50:.4f}")
    print(f"Box  mAP50-95: {metrics.box.map:.4f}")
    print(f"Mask mAP50:    {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
