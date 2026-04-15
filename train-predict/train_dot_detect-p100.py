"""
DOT 缺陷目标检测训练 (Ultralytics YOLO，多通道 .npy) - P100 兼容版本

数据为 (H,W,9) uint8；dataset.yaml 需含 channels: 9。首层卷积由 data['channels'] 构建，
加载预训练权重时对首层输入通道：对预训练 3 通道权重在通道维求均值后 broadcast 到全部通道（见 ultralytics-aoi BaseModel.load）。

优先使用仓库内 ultralytics-aoi 源码（与官方同步，含 .npy / 多通道补丁）。

针对 P100 (sm_60) 优化：关闭 AMP，默认 batch 减小，支持 CPU 训练。

用法:
    cd /roing-workspace/train-predict && python train_dot_detect-p100.py
    python train_dot_detect-p100.py --model yolo26m.pt --epochs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 使用工作区内的 ultralytics 源码（若存在）
_REPO = Path(__file__).resolve().parent.parent
_AOI = _REPO / "ultralytics-aoi"
if _AOI.is_dir():
    sys.path.insert(0, str(_AOI))

from ultralytics import YOLO

DATASET_YAML = "/roing-workspace/dataset/cleaned_YOLO_DOT/dataset.yaml"
DEFAULT_MODEL = "yolo26m.pt"
PROJECT_DIR = "/roing-workspace/train-predict/runs/dot_detect_p100"


def parse_args():
    p = argparse.ArgumentParser(description="DOT 多通道检测训练 (P100 兼容)")
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="预训练权重 .pt（首层 in_ch 与 channels 对齐；3→N 通道为 mean 后 broadcast）",
    )
    p.add_argument("--data", type=str, default=DATASET_YAML)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--batch",
        type=int,
        default=4,  # P100 显存小，减小 batch
        help="全局 batch（多卡时为所有 GPU 上的样本总和；P100 建议 4～8，OOM 再减）",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--device",
        type=str,
        default="0",  # P100 默认 GPU，如果不可用可设为 cpu
        help="设备：GPU '0,1,2,3'（DDP），单 GPU '0'，CPU 'cpu'（P100 sm_60 不兼容时用）",
    )
    p.add_argument(
        "--nbs",
        type=int,
        default=32,
        help="名义 batch（Ultralytics 用 nbs/batch 决定梯度累加步数；如 batch=4、nbs=32 → 约每 8 step 优化一次）",
    )
    p.add_argument("--no-amp", action="store_true", help="关闭自动混合精度（P100 无 Tensor Core，默认关闭）")
    p.add_argument("--workers", type=int, default=4)  # P100 CPU 弱，减小 workers
    p.add_argument("--name", type=str, default="exp", help="实验名")
    p.add_argument("--resume", action="store_true", help="从 checkpoint 继续")
    p.add_argument("--patience", type=int, default=100, help="EarlyStopping patience，0=禁用")
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
        resume_ckpt = None
        if Path(args.model).is_file():
            resume_ckpt = Path(args.model)
        else:
            checkpoints = sorted(Path(PROJECT_DIR).rglob("weights/last.pt"))
            if checkpoints:
                resume_ckpt = checkpoints[-1]
        if resume_ckpt is None:
            raise FileNotFoundError(
                f"未找到 checkpoint: {args.model} 或 {PROJECT_DIR}/weights/last.pt"
            )
        model = YOLO(str(resume_ckpt))
        # resume=True 会继续从 checkpoint 训练，checkpoint 中的首层通道已与 dataset.yaml 一致，
        # 不会再使用 3→N 通道的 mean broadcast 初始化逻辑。
        model.train(resume=True)
        return

    if not Path(args.model).is_file():
        raise FileNotFoundError(
            f"权重不存在: {args.model}\n请放置 yolo26m.pt 到本地或改用 --model yolo26m.pt 让 Ultralytics 自动下载"
        )

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        nbs=args.nbs,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        amp=False,  # P100 无 Tensor Core，强制关闭 AMP
        project=PROJECT_DIR,
        name=args.name,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=args.cos_lr,
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
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()