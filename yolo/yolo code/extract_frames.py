"""
extract_frames.py —— 视频自动抽帧

用法:
    python extract_frames.py clips/ball.mp4
    python extract_frames.py clips/ball.mp4 --out dataset/raw_frames --step 3
    python extract_frames.py clips/ball.mp4 --min-blur 0 --max-similarity 1.0

说明:
    - 每隔 --step 帧抽 1 张 (默认 3, 即每 3 帧留 1 帧)
    - --min-blur: 拉普拉斯方差低于该值的模糊帧丢弃 (设 0 关闭, 默认 40)
    - --max-similarity: 与上一张保存帧相似度高于该值的重复帧丢弃 (设 1 关闭, 默认 0.99)
    - 图片统一缩放到 --max-width 宽度 (默认 1280, 和传统视觉程序一致)
    - 输出到 --out 目录, 命名 000001.jpg, 000002.jpg ...
    - 同时生成 manifest.csv, 记录每一帧的来源视频帧号和时间

输出:
    dataset/raw_frames/
        000001.jpg
        000002.jpg
        ...
        manifest.csv
"""

import argparse
import csv
import os

import cv2


# ============================================================
#                    命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="视频自动抽帧")
    parser.add_argument("video", help="输入视频文件, 如 clips/ball.mp4")
    parser.add_argument(
        "--out", default="dataset/raw_frames",
        help="输出图片目录 (默认: dataset/raw_frames)"
    )
    parser.add_argument(
        "--step", type=int, default=3,
        help="每隔多少帧抽 1 张 (默认: 3)"
    )
    parser.add_argument(
        "--max-width", type=int, default=1280,
        help="图片最大宽度 (默认: 1280)"
    )
    parser.add_argument(
        "--min-blur", type=float, default=40.0,
        help="拉普拉斯方差阈值, 低于该值判定为模糊帧丢弃 (默认: 40, 设 0 关闭)"
    )
    parser.add_argument(
        "--max-similarity", type=float, default=0.99,
        help="与上一张保存帧相似度超过该值则丢弃 (默认: 0.99, 设 1 关闭)"
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=95,
        help="JPG 保存质量 (默认: 95)"
    )
    return parser.parse_args()


# ============================================================
#                    工具函数
# ============================================================

def resize_width(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(
        frame,
        (max_width, int(h * scale)),
        interpolation=cv2.INTER_AREA
    )


def blur_score(frame):
    """拉普拉斯方差, 值越小画面越模糊。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def thumbnail(frame, size=64):
    """把帧缩成小灰度图, 用于快速比较两帧是否几乎相同。"""
    small = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def frame_similarity(thumb_a, thumb_b):
    """0~1, 1 表示两帧完全相同。"""
    diff = cv2.absdiff(thumb_a, thumb_b)
    return 1.0 - float(diff.mean() / 255.0)


# ============================================================
#                     主程序
# ============================================================

def main():
    args = parse_args()

    if args.step < 1:
        print("--step 必须 >= 1")
        return

    if not os.path.exists(args.video):
        print(f"找不到视频: {args.video}")
        return

    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法打开视频: {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print("=" * 60)
    print("视频自动抽帧")
    print("=" * 60)
    print(f"视频: {args.video}")
    print(f"总帧数: {total_frames}")
    print(f"视频帧率: {video_fps:.1f} FPS")
    print(f"抽帧步长: 每 {args.step} 帧抽 1 张")
    print(f"输出目录: {args.out}")
    print()

    manifest_path = os.path.join(args.out, "manifest.csv")
    manifest = open(manifest_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(manifest)
    writer.writerow(["file", "src_frame", "time_sec", "blur", "width", "height"])

    saved_count = 0
    skipped_blur = 0
    skipped_similar = 0
    last_thumb = None

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 每隔 step 帧抽一帧
            if frame_idx % args.step != 0:
                frame_idx += 1
                continue

            frame = resize_width(frame, args.max_width)

            # 模糊检测
            blur = blur_score(frame)
            if args.min_blur > 0 and blur < args.min_blur:
                skipped_blur += 1
                frame_idx += 1
                continue

            # 重复帧检测
            if args.max_similarity < 1.0 and last_thumb is not None:
                sim = frame_similarity(last_thumb, thumbnail(frame))
                if sim > args.max_similarity:
                    skipped_similar += 1
                    frame_idx += 1
                    continue

            file_name = f"{saved_count + 1:06d}.jpg"
            file_path = os.path.join(args.out, file_name)

            h, w = frame.shape[:2]
            ok = cv2.imwrite(
                file_path, frame,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            )
            if not ok:
                print(f"写入失败: {file_path}")
                break

            time_sec = frame_idx / max(video_fps, 1e-6)
            writer.writerow([
                file_name, frame_idx, f"{time_sec:.3f}",
                f"{blur:.1f}", w, h
            ])

            last_thumb = thumbnail(frame)
            saved_count += 1

            if saved_count % 50 == 0:
                print(f"  已保存 {saved_count} 张 ...")

            frame_idx += 1
    finally:
        cap.release()
        manifest.close()

    print()
    print("-" * 60)
    print(f"完成! 保存 {saved_count} 张图片")
    print(f"丢弃模糊帧: {skipped_blur}")
    print(f"丢弃重复帧: {skipped_similar}")
    print(f"图片目录: {args.out}")
    print(f"清单文件: {manifest_path}")
    print("-" * 60)
    print("下一步: python auto_label.py --frames " + args.out)


if __name__ == "__main__":
    main()
