# -*- coding: utf-8 -*-

import cv2
import numpy as np


# ============================================================
# 1. DroidCam
# ============================================================

DROIDCAM_IP = "172.30.230.152"

VIDEO_URL = f"http://{DROIDCAM_IP}:4747/video"


# ============================================================
# 2. 打开摄像头
# ============================================================

cap = cv2.VideoCapture(VIDEO_URL)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():

    print("❌ 无法打开 DroidCam")
    print("请检查：")
    print("1. 手机 DroidCam 是否打开")
    print("2. IP 是否正确")
    print("3. 电脑和手机是否在同一个网络")
    exit()


print("====================================")
print("HSV 取色调试器")
print("====================================")
print("鼠标左键：点击取色")
print("Q：退出")
print("====================================")


# ============================================================
# 3. 保存鼠标点击位置
# ============================================================

click_points = []


# ============================================================
# 4. 鼠标回调函数
# ============================================================

def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        frame = param

        # 防止坐标越界
        if (
            y < 0
            or y >= frame.shape[0]
            or x < 0
            or x >= frame.shape[1]
        ):
            return

        # BGR
        b, g, r = frame[y, x]

        # HSV
        hsv_pixel = cv2.cvtColor(
            np.uint8([[[b, g, r]]]),
            cv2.COLOR_BGR2HSV
        )

        h, s, v = hsv_pixel[0, 0]

        # 保存点击点
        click_points.append(
            {
                "x": x,
                "y": y,
                "bgr": (int(b), int(g), int(r)),
                "hsv": (int(h), int(s), int(v))
            }
        )

        print()
        print("====================================")
        print(f"点击位置：({x}, {y})")
        print("------------------------------------")
        print(f"BGR = ({b}, {g}, {r})")
        print(f"HSV = ({h}, {s}, {v})")
        print(f"H = {h}")
        print(f"S = {s}")
        print(f"V = {v}")
        print("====================================")


# ============================================================
# 5. 主循环
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret or frame is None:

        print("❌ 无法读取视频帧")
        break


    # --------------------------------------------------------
    # 创建窗口
    # --------------------------------------------------------

    cv2.namedWindow(
        "HSV Picker"
    )

    cv2.setMouseCallback(
        "HSV Picker",
        mouse_callback,
        frame
    )


    # --------------------------------------------------------
    # 在画面上显示之前点击的位置
    # --------------------------------------------------------

    display = frame.copy()

    for point in click_points:

        x = point["x"]
        y = point["y"]

        h, s, v = point["hsv"]

        # 画十字
        cv2.drawMarker(
            display,
            (x, y),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            20,
            2
        )

        # 显示 HSV
        text = f"H:{h} S:{s} V:{v}"

        cv2.putText(
            display,
            text,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    # --------------------------------------------------------
    # 操作提示
    # --------------------------------------------------------

    cv2.putText(
        display,
        "Click = HSV | Q = Quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # 显示
    # --------------------------------------------------------

    cv2.imshow(
        "HSV Picker",
        display
    )


    # --------------------------------------------------------
    # 键盘
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# 6. 释放
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("程序结束。")