"""
collect_video.py —— 从手机 IP 摄像头采集钢球运动视频

用法:
    python collect_video.py                          # 默认采集 30 秒
    python collect_video.py --duration 20            # 采集 20 秒
    python collect_video.py --out clips/ball_01.mp4  # 指定输出文件名
    python collect_video.py --ip 172.30.230.152 --port 4747

说明:
    - 连接手机 IP 摄像头 (DroidCam / IP Webcam 等 MJPEG 视频流)
    - 边采集边显示实时画面
    - 按 Q 提前停止, 或到达 --duration 秒自动停止
    - 视频保存为 mp4, 供 extract_frames.py 抽帧

拍摄建议:
    - 让钢球 左->中->右, 右->中->左, 快->慢, 尽量把比赛可能出现的状态都拍进去
    - 手机固定, 不要晃动, 绿色管道尽量完整出现在画面里
    - 20~30 秒就够, 不需要拍几百张照片
"""

import argparse
import os
import time

import cv2


# ============================================================
#                    命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="从手机 IP 摄像头采集钢球运动视频"
    )
    parser.add_argument(
        "--ip", default="172.30.230.152",
        help="手机 IP 摄像头地址 (默认: 172.30.230.152)"
    )
    parser.add_argument(
        "--port", type=int, default=4747,
        help="端口 (默认: 4747)"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="采集时长秒数 (默认: 30, 按 Q 可提前停止)"
    )
    parser.add_argument(
        "--out", default="clips/ball.mp4",
        help="输出视频文件路径 (默认: clips/ball.mp4)"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="写视频的帧率 (默认: 30, 只影响播放速度标注)"
    )
    parser.add_argument(
        "--max-width", type=int, default=1280,
        help="保存画面的最大宽度 (默认: 1280, 和传统视觉一致)"
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


def fmt_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


# ============================================================
#                     主程序
# ============================================================

def main():
    args = parse_args()
    video_url = f"http://{args.ip}:{args.port}/video"

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("手机摄像头视频采集")
    print("=" * 60)
    print(f"视频地址: {video_url}")
    print(f"目标时长: {args.duration:.1f} 秒 (按 Q 提前停止)")
    print(f"输出文件: {args.out}")
    print()

    cap = cv2.VideoCapture(video_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("手机摄像头连接失败!")
        print("请检查: 手机 IP 是否正确, DroidCam 是否打开, 手机和电脑是否同一 Wi-Fi")
        return

    # 先读一帧, 确定画面尺寸
    ret, frame = cap.read()
    if not ret:
        print("无法从摄像头读取画面!")
        cap.release()
        return

    frame = resize_width(frame, args.max_width)
    h, w = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
    if not writer.isOpened():
        print(f"无法写入视频文件: {args.out}")
        cap.release()
        return

    print(f"画面尺寸: {w} x {h}")
    print("REC 开始! 让钢球运动起来...")
    print()

    start = time.perf_counter()
    frames_written = 0
    rec_printed = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # 偶尔丢一帧, 跳过即可
                if time.perf_counter() - start > args.duration + 5:
                    break
                continue

            frame = resize_width(frame, args.max_width)
            writer.write(frame)
            frames_written += 1

            elapsed = time.perf_counter() - start

            # 实时预览
            display = frame.copy()
            cv2.rectangle(display, (0, 0), (display.shape[1], 70), (0, 0, 0), -1)
            cv2.putText(
                display,
                f"REC {fmt_time(elapsed)} / {fmt_time(args.duration)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            cv2.putText(
                display,
                f"Frames: {frames_written}   Q = 停止",
                (15, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
            )
            cv2.imshow("Recording (Q to stop)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[USER] 手动停止采集")
                break

            if elapsed >= args.duration:
                print("到达目标时长, 自动停止")
                break
    finally:
        writer.release()
        cap.release()
        cv2.destroyAllWindows()

    total_time = time.perf_counter() - start
    actual_fps = frames_written / max(total_time, 1e-6)

    print()
    print("-" * 60)
    print(f"采集完成: {args.out}")
    print(f"实际时长: {total_time:.1f} 秒")
    print(f"保存帧数: {frames_written}")
    print(f"平均帧率: {actual_fps:.1f} FPS")
    print("-" * 60)
    print("下一步: python extract_frames.py " + args.out)


if __name__ == "__main__":
    main()
