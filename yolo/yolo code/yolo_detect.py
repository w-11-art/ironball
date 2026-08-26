"""
yolo_detect.py —— 实时 YOLO 钢球检测

钢球识别交给 YOLO, 管道识别复用传统视觉程序:
    手机摄像头 -> YOLO 找球 + 传统视觉找管道
              -> 球心投影到管道轴线 -> 换算成 cm 距离

画面显示:
    红框 + 中心点 + 置信度        (YOLO 钢球)
    绿框 + 中心十字线             (传统视觉管道)
    Pos: +x.xx cm                球沿管道方向的位置 (管道中心 = 0)
    |Center|: x.xx cm            球距离管道中心的距离
    CAM: xx.x FPS                摄像头真实帧率
    DET: xx.x FPS                处理/检测帧率

用法:
    python yolo_detect.py                          # 手机摄像头实时检测
    python yolo_detect.py --source clips/ball.mp4  # 检测视频文件
    python yolo_detect.py --source image/xx.jpg    # 检测单张图片

按键 (摄像头/视频模式):
    Q  退出
    S  保存当前画面到 captures/

参数:
    --model    模型路径 (默认 models/ball_yolo.pt)
    --conf     置信度阈值 (默认 0.50)
    --imgsz    YOLO 输入尺寸 (默认 640)
    --ip/--port 手机 IP 摄像头地址
    --pipe-interval  每 N 帧重新识别一次管道 (默认 5)
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    print("=" * 60)
    print("[错误] 没找到 ultralytics 模块!")
    print()
    print("原因: 大概率用错了 Python 解释器 (例如系统 Python 3.11,")
    print("      而不是 conda 的 yolo 环境)。")
    print()
    print("正确做法 (PowerShell):")
    print("    conda activate yolo")
    print("    python yolo_detect.py")
    print()
    print("或者直接用项目里的启动器:")
    print("    run.bat yolo_detect.py")
    print("=" * 60)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ============================================================
#          加载传统视觉程序 (复用摄像头类 + 管道识别)
# ============================================================

_spec = importlib.util.spec_from_file_location(
    "recognition_final", ROOT / "codes" / "recognition final.py"
)
rv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rv)

LatestFrameCamera = rv.LatestFrameCamera
find_pipe_model = rv.find_pipe_model
pipe_model_from_box = rv.pipe_model_from_box
point_to_position_cm = rv.point_to_position_cm
PIPE_LENGTH_CM = rv.PIPE_LENGTH_CM


# ============================================================
#                    命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="实时 YOLO 钢球检测")
    parser.add_argument(
        "--source", default="",
        help="输入来源: 空=手机摄像头, 视频文件路径, 或图片路径"
    )
    parser.add_argument("--ip", default="172.30.230.152", help="手机 IP 摄像头地址")
    parser.add_argument("--port", type=int, default=4747, help="端口")
    parser.add_argument(
        "--model", default="models/ball_yolo.pt",
        help="YOLO 模型路径 (默认: models/ball_yolo.pt)"
    )
    parser.add_argument("--conf", type=float, default=0.50, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 输入尺寸")
    parser.add_argument("--max-width", type=int, default=1280, help="画面最大宽度")
    parser.add_argument("--save-dir", default="captures", help="S 键保存目录")
    parser.add_argument(
        "--pipe-interval", type=int, default=5,
        help="每 N 帧重新识别一次管道 (默认: 5, 管道静止, 不用每帧找)"
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


def draw_detection(frame, box, conf):
    """画 YOLO 钢球框 + 中心点 + 置信度。"""
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    cv2.putText(
        frame, f"ball {conf:.2f}",
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
    )
    cv2.putText(
        frame, f"({cx}, {cy})",
        (cx + 8, cy - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
    )


def draw_pipe(frame, pipe_model):
    """画管道框 + 中心十字线, 让用户直观看到管道中心在哪。"""
    if pipe_model is None:
        return

    import numpy as np
    pts = rv.order_box_points(pipe_model["box"]).astype(np.int32)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    center = pipe_model["center"]
    direction = pipe_model["direction"]
    half_len = pipe_model["length_px"] / 2.0

    left = center - direction * half_len
    right = center + direction * half_len

    cv2.line(
        frame,
        tuple(np.round(left).astype(int)),
        tuple(np.round(right).astype(int)),
        (0, 255, 0), 1
    )

    cx, cy = int(round(center[0])), int(round(center[1]))
    cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (0, 255, 255), 2)
    cv2.line(frame, (cx, cy - 12), (cx, cy + 12), (0, 255, 255), 2)
    cv2.putText(
        frame, "CENTER", (cx + 16, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1
    )


def detect_image(model, frame, conf, imgsz):
    """对单帧做 YOLO 检测, 返回 (画面, 球中心点或 None)。"""
    results = model.predict(
        frame, conf=conf, imgsz=imgsz, verbose=False
    )[0]

    ball_center = None
    best_conf = -1.0

    for b in results.boxes:
        box = b.xyxy[0].tolist()
        box_conf = float(b.conf[0])
        draw_detection(frame, box, box_conf)

        if box_conf > best_conf:
            best_conf = box_conf
            ball_center = (
                (box[0] + box[2]) / 2.0,
                (box[1] + box[3]) / 2.0,
            )

    return frame, ball_center


def ball_distance_cm(ball_center, pipe_model):
    """球心沿管道轴线的位置 (cm), 管道中心 = 0。"""
    if ball_center is None or pipe_model is None:
        return None
    return point_to_position_cm(ball_center, pipe_model)


# ============================================================
#                    摄像头实时检测
# ============================================================

def run_camera_loop(model, args, source_path=None):
    """手机摄像头 / 视频文件 实时检测。"""
    os.makedirs(args.save_dir, exist_ok=True)

    if source_path is None:
        url = f"http://{args.ip}:{args.port}/video"
        print(f"连接手机摄像头: {url}")
        camera = LatestFrameCamera(url)
        time.sleep(1.5)
        if not camera.cap.isOpened():
            print("手机摄像头连接失败!")
            camera.stop()
            return
        print("连接成功!")
    else:
        camera = cv2.VideoCapture(source_path)
        if not camera.isOpened():
            print(f"无法打开视频: {source_path}")
            return

    # 管道状态 (静止, 缓存 + 每 N 帧更新)
    pipe_model = None
    pipe_fail_count = 0
    pipe_lost_limit = 10
    frame_counter = 0

    # 帧率统计
    camera_fps = 0.0
    camera_count = 0
    last_frame_id = -1
    camera_start = time.perf_counter()

    proc_fps = 0.0
    proc_count = 0
    proc_start = time.perf_counter()

    try:
        while True:
            # ---------------- 读取帧 ----------------
            if source_path is None:
                frame, frame_id = camera.read()
            else:
                ret, frame = camera.read()
                if not ret:
                    break
                frame_id = -1

            if frame is None:
                continue

            frame = resize_width(frame, args.max_width)
            now = time.perf_counter()
            frame_counter += 1

            # ---------------- CAM FPS (真实帧率) ----------------
            if source_path is None:
                # 手机摄像头: 用 frame_id 的跳变来数实际到达的帧
                if last_frame_id < 0:
                    last_frame_id = frame_id
                elif frame_id != last_frame_id:
                    camera_count += frame_id - last_frame_id
                    last_frame_id = frame_id
            else:
                # 视频文件: 每读一帧就是实际一帧
                camera_count += 1

            if now - camera_start >= 1.0:
                camera_fps = camera_count / (now - camera_start)
                camera_count = 0
                camera_start = now

            # ---------------- PROC FPS ----------------
            proc_count += 1
            if now - proc_start >= 1.0:
                proc_fps = proc_count / (now - proc_start)
                proc_count = 0
                proc_start = now

            # ---------------- 管道识别 (每 N 帧) ----------------
            need_pipe = (
                pipe_model is None
                or frame_counter % max(args.pipe_interval, 1) == 0
            )
            if need_pipe:
                pipe_info, _debug = find_pipe_model(frame)
                if pipe_info is not None:
                    pipe_model = pipe_model_from_box(pipe_info["box"])
                    pipe_fail_count = 0
                else:
                    pipe_fail_count += 1
                    if pipe_fail_count > pipe_lost_limit:
                        pipe_model = None

            # ---------------- YOLO 检测 ----------------
            frame, ball_center = detect_image(
                model, frame, args.conf, args.imgsz
            )

            # ---------------- 管道绘制 ----------------
            draw_pipe(frame, pipe_model)

            # ---------------- 距离换算 ----------------
            pos_cm = ball_distance_cm(ball_center, pipe_model)
            has_ball = ball_center is not None

            # ---------------- 文字叠加 ----------------
            status_color = (0, 255, 0) if has_ball else (0, 0, 255)
            cv2.putText(
                frame,
                f"YOLO: {'BALL' if has_ball else 'SEARCHING'}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
            )

            if pos_cm is not None:
                cv2.putText(
                    frame,
                    f"Pos: {pos_cm:+.2f} cm   |Center|: {abs(pos_cm):.2f} cm",
                    (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
            elif has_ball:
                cv2.putText(
                    frame, "Pos: -- (pipe not found)",
                    (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1
                )
            else:
                cv2.putText(
                    frame, "Pos: -- (no ball)",
                    (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1
                )

            # 右上角: 两个真实帧率
            cv2.putText(
                frame, f"CAM: {camera_fps:.1f} FPS",
                (frame.shape[1] - 210, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"DET: {proc_fps:.1f} FPS",
                (frame.shape[1] - 210, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            cv2.putText(
                frame, f"Pipe: {PIPE_LENGTH_CM:.0f} cm  Q=quit  S=save",
                (15, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

            # ---------------- 显示 ----------------
            cv2.imshow("YOLO Ball Detection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[USER] 退出")
                break

            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(args.save_dir, f"ball_{stamp}.jpg")
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"[SAVE] {path}")

    finally:
        if source_path is None:
            camera.stop()
        else:
            camera.release()
        cv2.destroyAllWindows()


# ============================================================
#                     主程序
# ============================================================

def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"找不到模型: {args.model}")
        print("请先运行: python train_yolo.py")
        return

    model = YOLO(args.model)
    print("=" * 60)
    print("YOLO 钢球实时检测 (管道总长 %.0f cm)" % PIPE_LENGTH_CM)
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"置信度阈值: {args.conf}")
    print()

    if args.source:
        # 单张图片模式
        if args.source.lower().endswith(IMAGE_EXTS):
            print(f"检测单张图片: {args.source}")
            frame = cv2.imread(args.source)
            if frame is None:
                print("图片读取失败!")
                return
            frame = resize_width(frame, args.max_width)

            # 管道识别
            pipe_info, _debug = find_pipe_model(frame)
            pipe_model = None
            if pipe_info is not None:
                pipe_model = pipe_model_from_box(pipe_info["box"])
                print("管道识别成功")

            # YOLO 检测
            frame, ball_center = detect_image(
                model, frame, args.conf, args.imgsz
            )
            draw_pipe(frame, pipe_model)

            pos_cm = ball_distance_cm(ball_center, pipe_model)
            if pos_cm is not None:
                print(f"球位置: {pos_cm:+.2f} cm, 距中心: {abs(pos_cm):.2f} cm")
            else:
                print("球位置: 未检测到球或未找到管道")

            cv2.putText(
                frame,
                f"YOLO: {'BALL' if ball_center is not None else 'NOT FOUND'}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if ball_center is not None else (0, 0, 255), 2
            )
            cv2.imshow("YOLO Ball Detection", frame)
            print("按任意键退出")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("完成")
            return

        # 视频文件模式
        print(f"检测视频: {args.source}")
        run_camera_loop(model, args, source_path=args.source)

    else:
        # 手机摄像头模式
        run_camera_loop(model, args, source_path=None)


if __name__ == "__main__":
    main()
