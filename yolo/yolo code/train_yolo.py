"""
train_yolo.py —— 训练钢球检测 YOLO 模型

用法:
    python train_yolo.py                          # 默认 100 轮
    python train_yolo.py --epochs 150 --batch 16  # 自定义
    python train_yolo.py --model yolo11n.pt       # 换预训练模型

流程:
    1. 把 dataset/images + dataset/labels 按比例分成 train / val
    2. 生成 dataset/data.yaml
    3. 用预训练模型做迁移学习, 训练单类钢球检测
    4. 训练完把 best.pt 复制到 models/ball_yolo.pt

输出:
    models/ball_yolo.pt     训练好的钢球检测模型
    runs/ball_train/        训练日志、权重和曲线图

数据要求 (由 auto_label.py 生成):
    dataset/images/*.jpg   图片
    dataset/labels/*.txt   YOLO 标签 (0 cx cy w h)
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
IMAGES_DIR = DATASET / "images"
LABELS_DIR = DATASET / "labels"

CLASS_ID = 0
CLASS_NAME = "ball"


# ============================================================
#                    命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="训练钢球检测 YOLO")
    parser.add_argument(
        "--model", default="yolo26n.pt",
        help="预训练模型 (默认: yolo26n.pt, 其他可选 yolo11n.pt / yolov8n.pt)"
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="训练图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="batch 大小")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--patience", type=int, default=30, help="早停轮数")
    parser.add_argument("--device", default="0", help="训练设备, 0=GPU / cpu")
    parser.add_argument("--out-name", default="ball_train", help="runs/ 下的输出目录名")
    return parser.parse_args()


# ============================================================
#              划分数据集 + 生成 data.yaml
# ============================================================

def prepare_dataset(args):
    """把 dataset/images + dataset/labels 划分成 train/val, 生成 data.yaml。"""
    train_img = DATASET / "train" / "images"
    train_lab = DATASET / "train" / "labels"
    val_img = DATASET / "val" / "images"
    val_lab = DATASET / "val" / "labels"

    for d in [train_img, train_lab, val_img, val_lab]:
        d.mkdir(parents=True, exist_ok=True)

    # 只保留图片和标签都存在的样本
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    pairs = []
    for img in images:
        lab = LABELS_DIR / (img.stem + ".txt")
        if lab.exists():
            pairs.append((img, lab))

    if not pairs:
        print("没有找到成对的 图片+标签, 请先运行 auto_label.py")
        raise SystemExit(1)

    random.seed(args.seed)
    random.shuffle(pairs)

    val_count = max(1, int(len(pairs) * args.val_ratio))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    for img, lab in train_pairs:
        shutil.copy2(img, train_img / img.name)
        shutil.copy2(lab, train_lab / lab.name)

    for img, lab in val_pairs:
        shutil.copy2(img, val_img / img.name)
        shutil.copy2(lab, val_lab / lab.name)

    # 生成 data.yaml (用绝对路径, 避免找不到文件)
    data_yaml = {
        "path": str(DATASET.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {CLASS_ID: CLASS_NAME},
    }
    yaml_path = DATASET / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    return yaml_path, len(train_pairs), len(val_pairs)


# ============================================================
#                     主程序
# ============================================================

def main():
    args = parse_args()

    if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
        print("找不到 dataset/images 或 dataset/labels")
        print("请先运行: python auto_label.py")
        return

    print("=" * 60)
    print("YOLO 钢球检测训练")
    print("=" * 60)

    yaml_path, n_train, n_val = prepare_dataset(args)
    print(f"训练集: {n_train} 张")
    print(f"验证集: {n_val} 张")
    print(f"data.yaml: {yaml_path}")
    print(f"预训练模型: {args.model}")
    print(f"epochs={args.epochs}  imgsz={args.imgsz}  batch={args.batch}")
    print("开始训练 (按 Ctrl+C 可停止)...")
    print()

    # 加载预训练模型并训练
    model = YOLO(args.model)

    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.out_name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    # 把 best.pt 复制到 models/
    best_pt = ROOT / "runs" / args.out_name / "weights" / "best.pt"
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    dst = models_dir / "ball_yolo.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, dst)
        print()
        print("=" * 60)
        print(f"训练完成! 模型已保存:")
        print(f"  {dst}")
        print(f"  全部权重: {ROOT / 'runs' / args.out_name / 'weights'}")
        print(f"  训练曲线: {ROOT / 'runs' / args.out_name}")
        print()
        print("下一步: python yolo_detect.py  (实时 YOLO 检测)")
    else:
        print(f"[警告] 没找到 {best_pt}, 训练可能中断")


if __name__ == "__main__":
    main()
