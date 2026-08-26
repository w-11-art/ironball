"""
auto_label.py —— 用传统视觉程序自动生成 YOLO 钢球标签

这是你整套流程里"最省事"的一步:
    拍一小段视频 -> 抽帧 -> 传统视觉自动框钢球 -> 生成 YOLO 标签

用法:
    python auto_label.py --frames dataset/raw_frames
    python auto_label.py --frames dataset/raw_frames --min-score 0.55
    python auto_label.py --frames dataset/raw_frames --keep-empty

原理:
    对每张抽帧图片, 直接复用 codes/recognition final.py 里已经调好的逻辑:
    1. 自动识别绿色管道        -> find_pipe_model
    2. 锁定管道内钢球          -> find_ball_candidates
    3. 取分数最高的候选, 把 bbox 换算成 YOLO 格式
    4. 写入 001.txt:  0 cx cy w h (全部归一化 0~1)

输出:
    dataset/images/  001.jpg ...   有钢球的帧 (原图, 可直接喂给 YOLO)
    dataset/labels/  001.txt ...   YOLO 标签
    dataset/review/  001_box.jpg   带框预览图, 供人工快速检查
    dataset/report.csv             每帧检测结果汇总

注意:
    生成后请务必人工检查 review/ 下的预览图:
    框错了 / 漏球 / 把反光当球 -> 用 LabelImg 打开 dataset/images 修正,
    或直接删掉对应的 jpg + txt。
"""

import argparse
import csv
import importlib.util
import os
import sys

import cv2


# ============================================================
#              加载传统视觉程序 (codes/recognition final.py)
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_FILE = os.path.join(ROOT_DIR, "codes", "recognition final.py")

_spec = importlib.util.spec_from_file_location("recognition_final", CODE_FILE)
rv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rv)

find_pipe_model = rv.find_pipe_model
pipe_model_from_box = rv.pipe_model_from_box
find_ball_candidates = rv.find_ball_candidates
PROCESS_WIDTH = rv.PROCESS_WIDTH

# 检测参数都是按 1280 宽度调好的, 一律缩放到该宽度再处理
# (extract_frames.py 默认抽帧宽度也是 1280, 二者保持一致)

CLASS_ID = 0          # YOLO 类别 0 = 钢球
CLASS_NAME = "ball"   # data.yaml 里用


# ============================================================
#                    命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="用传统视觉程序自动生成 YOLO 钢球标签"
    )
    parser.add_argument(
        "--frames", default="dataset/raw_frames",
        help="抽帧图片目录 (默认: dataset/raw_frames)"
    )
    parser.add_argument(
        "--out-images", default="dataset/images",
        help="输出 YOLO 图片目录 (默认: dataset/images)"
    )
    parser.add_argument(
        "--out-labels", default="dataset/labels",
        help="输出 YOLO 标签目录 (默认: dataset/labels)"
    )
    parser.add_argument(
        "--out-review", default="dataset/review",
        help="带框预览图目录 (默认: dataset/review)"
    )
    parser.add_argument(
        "--out-report", default="dataset/report.csv",
        help="检测结果汇总 CSV (默认: dataset/report.csv)"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.50,
        help="钢球候选最低分数, 低于该值不生成标签 (默认: 0.50)"
    )
    parser.add_argument(
        "--keep-empty", action="store_true",
        help="把'管道正常但没球'的帧也保存为负样本 (空标签, 减少误检)"
    )
    return parser.parse_args()


# ============================================================
#                    工具函数
# ============================================================

def bbox_to_yolo(bbox, img_w, img_h):
    """把 (x1, y1, w, h) 像素框转成 YOLO 归一化 (cx, cy, w, h)。"""
    x1, y1, w, h = bbox

    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(x1 + 1, min(int(x1 + w), img_w))
    y2 = max(y1 + 1, min(int(y1 + h), img_h))

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    nw = (x2 - x1) / img_w
    nh = (y2 - y1) / img_h

    return cx, cy, nw, nh


def draw_ball_box(image, bbox, score, color=(0, 0, 255)):
    x1, y1, w, h = bbox
    cv2.rectangle(
        image,
        (int(x1), int(y1)),
        (int(x1 + w), int(y1 + h)),
        color, 2
    )
    label = f"ball {score:.2f}"
    cv2.putText(
        image, label,
        (int(x1), max(16, int(y1) - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
    )


def main():
    args = parse_args()

    if not os.path.isdir(args.frames):
        print(f"找不到抽帧目录: {args.frames}")
        print("先运行 extract_frames.py 生成抽帧, 再运行本程序")
        sys.exit(1)

    os.makedirs(args.out_images, exist_ok=True)
    os.makedirs(args.out_labels, exist_ok=True)
    os.makedirs(args.out_review, exist_ok=True)

    image_exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted(
        f for f in os.listdir(args.frames)
        if f.lower().endswith(image_exts)
    )

    if not files:
        print(f"目录 {args.frames} 里没有图片")
        sys.exit(1)

    print("=" * 60)
    print("传统视觉自动标注 YOLO 钢球")
    print("=" * 60)
    print(f"图片数量: {len(files)}")
    print(f"最低分数: {args.min_score}")
    print(f"输出图片: {args.out_images}")
    print(f"输出标签: {args.out_labels}")
    print(f"预览图  : {args.out_review}")
    print(f"报告    : {args.out_report}")
    print()

    report_path = args.out_report
    report = open(report_path, "w", newline="", encoding="utf-8")
    report_writer = csv.writer(report)
    report_writer.writerow([
        "image", "status", "has_pipe", "ball_count",
        "best_score", "cx", "cy", "radius",
        "bbox_x1", "bbox_y1", "bbox_w", "bbox_h"
    ])

    stats = {
        "labeled": 0,
        "no_pipe": 0,
        "no_ball": 0,
        "low_score": 0,
        "empty_kept": 0,
        "failed": 0,
    }

    for index, file_name in enumerate(files):
        src_path = os.path.join(args.frames, file_name)
        stem = os.path.splitext(file_name)[0]

        frame = cv2.imread(src_path)
        if frame is None:
            print(f"[SKIP] 无法读取: {file_name}")
            stats["failed"] += 1
            report_writer.writerow([file_name, "failed", "", "", "", "", "", "", "", "", "", ""])
            continue

        frame = rv.resize_frame(frame)
        img_h, img_w = frame.shape[:2]

        # ---------- 1. 自动识别绿色管道 ----------
        pipe_info, pipe_debug = find_pipe_model(frame)
        if pipe_info is None:
            stats["no_pipe"] += 1
            report_writer.writerow([file_name, "no_pipe", "0", "", "", "", "", "", "", "", "", ""])
            continue

        pipe_model = pipe_model_from_box(pipe_info["box"])

        # ---------- 2. 锁定管道内钢球 ----------
        candidates, scene_info = find_ball_candidates(frame, pipe_model)
        no_ball_uniform_green = scene_info["no_ball_uniform_green"]

        if no_ball_uniform_green or not candidates:
            if no_ball_uniform_green and args.keep_empty:
                # 负样本: 管道正常, 确实没有球
                # 保存处理后的帧, 保证和标签坐标系一致
                cv2.imwrite(
                    os.path.join(args.out_images, file_name), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                empty_txt = os.path.join(args.out_labels, stem + ".txt")
                with open(empty_txt, "w", encoding="utf-8") as f:
                    f.write("")  # 空标签 = 这张图里没有目标
                stats["empty_kept"] += 1
                status = "empty"
            else:
                status = "no_ball"
                stats["no_ball"] += 1
            report_writer.writerow([
                file_name, status, "1", str(len(candidates)),
                "", "", "", "", "", "", "", ""
            ])
            continue

        # ---------- 3. 取分数最高的候选 ----------
        best = candidates[0]
        if best["score"] < args.min_score:
            stats["low_score"] += 1
            report_writer.writerow([
                file_name, "low_score", "1", str(len(candidates)),
                f"{best['score']:.3f}",
                f"{best['x']:.1f}", f"{best['y']:.1f}", f"{best['radius']:.1f}",
                "", "", "", ""
            ])
            continue

        # ---------- 4. 生成 YOLO 标签 ----------
        bbox = best.get("bbox")
        if bbox is None:
            # 兜底: 用中心 + 半径构造方形框
            r = best["radius"]
            bbox = (best["x"] - r, best["y"] - r, 2 * r, 2 * r)

        cx, cy, nw, nh = bbox_to_yolo(bbox, img_w, img_h)

        label_txt = (
            f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"
        )
        txt_path = os.path.join(args.out_labels, stem + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(label_txt)

        # ---------- 5. 图片 + 预览图 ----------
        # 保存处理后的帧, 保证和标签坐标系一致
        cv2.imwrite(
            os.path.join(args.out_images, file_name), frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        review = frame.copy()
        draw_ball_box(review, bbox, best["score"])
        review_path = os.path.join(args.out_review, stem + "_box.jpg")
        cv2.imwrite(review_path, review, [cv2.IMWRITE_JPEG_QUALITY, 90])

        stats["labeled"] += 1

        report_writer.writerow([
            file_name, "labeled", "1", str(len(candidates)),
            f"{best['score']:.3f}",
            f"{best['x']:.1f}", f"{best['y']:.1f}", f"{best['radius']:.1f}",
            f"{bbox[0]:.1f}", f"{bbox[1]:.1f}",
            f"{bbox[2]:.1f}", f"{bbox[3]:.1f}"
        ])

        if (index + 1) % 50 == 0:
            print(f"  已处理 {index + 1}/{len(files)} ...")

    report.close()

    print()
    print("-" * 60)
    print(f"处理完成, 共 {len(files)} 张图片:")
    print(f"  自动标注      : {stats['labeled']}")
    if args.keep_empty:
        print(f"  空管负样本    : {stats['empty_kept']}")
    print(f"  找不到管道    : {stats['no_pipe']}")
    print(f"  管道内无球    : {stats['no_ball']}")
    print(f"  分数过低(疑似): {stats['low_score']}")
    print(f"  读取失败      : {stats['failed']}")
    print(f"  YOLO 图片目录 : {args.out_images}")
    print(f"  YOLO 标签目录 : {args.out_labels}")
    print(f"  预览图目录    : {args.out_review}")
    print(f"  检测报告      : {args.out_report}")
    print()
    print("!! 重要: 请先打开 review/ 目录人工检查每一张预览图")
    print("   框错/漏球/反光误判的, 用 LabelImg 打开 dataset/images 修正,")
    print("   或者直接删除对应的 jpg 和 txt。检查无误后再训练 YOLO。")
    print("-" * 60)


if __name__ == "__main__":
    main()
