import cv2
import numpy as np
import time
import threading
import math


# ============================================================
#                    1. 手机摄像头
# ============================================================

PHONE_IP = "172.30.230.152"
PHONE_PORT = 4747

VIDEO_URL = f"http://{PHONE_IP}:{PHONE_PORT}/video"


# ============================================================
#                    2. 管道参数
# ============================================================

# 管道实际长度
PIPE_LENGTH_CM = 25.0

# 管道在画面中的固定位置
#
# 你已经可以固定手机和管道，
# 所以这里直接固定。
#
PIPE_X1 = 120
PIPE_X2 = 1160

PIPE_Y_TOP = 300
PIPE_Y_BOTTOM = 430

PIPE_Y_CENTER = (
    PIPE_Y_TOP + PIPE_Y_BOTTOM
) // 2


# ============================================================
#              3. 管道 HSV
#
# !!! 这里的 HSV 只用于识别绿色管道 !!!
#
# 钢球完全不使用 HSV 判断。
#
# 先用这一组，后面根据你的实际绿色管道
# 可以继续微调。
# ============================================================

PIPE_H_LOW = 30
PIPE_H_HIGH = 95

PIPE_S_LOW = 45
PIPE_S_HIGH = 255

PIPE_V_LOW = 20
PIPE_V_HIGH = 255


# ============================================================
#                    4. 钢球大小
#
# 这里不判断钢球颜色。
#
# 只判断：
#     面积
#     半径
#     圆形度
#     宽高比
#     是否位于管道内部
#
# ============================================================

MIN_BALL_AREA = 30
MAX_BALL_AREA = 1800

MIN_BALL_RADIUS = 4
MAX_BALL_RADIUS = 35

MIN_CIRCULARITY = 0.35

MIN_ASPECT_RATIO = 0.55
MAX_ASPECT_RATIO = 1.80


# ============================================================
#              5. 钢球允许的竖直位置
#
# 钢球只会在管道内部。
#
# ============================================================

BALL_ZONE_TOP_RATIO = 0.15
BALL_ZONE_BOTTOM_RATIO = 0.85


# ============================================================
#                    6. 锁定参数
# ============================================================

# 已经锁定以后，只在附近寻找
TRACK_SEARCH_WIDTH = 160
TRACK_SEARCH_HEIGHT = 100

# 短暂丢球时扩大搜索
LOST_SEARCH_WIDTH = 260
LOST_SEARCH_HEIGHT = 130

# 连续多少帧找不到才重新全范围搜索
MAX_LOST_FRAMES = 8


# ============================================================
#                    7. 防止钢球跳点
# ============================================================

# 单帧最大允许移动
#
# 这是防止误识别，不是速度上限。
#
MAX_SINGLE_JUMP_CM = 2.0

# 最大合理速度
#
# 如果突然超过这个速度，
# 程序不会直接接受新目标。
MAX_REASONABLE_SPEED_CM_S = 100.0


# 自检需要连续多少帧确认
VERIFY_FRAMES = 2


# ============================================================
#                    8. 速度滤波
# ============================================================

VELOCITY_ALPHA = 0.25


# ============================================================
#                9. 真实 FPS
# ============================================================

PROCESS_WIDTH = 1280


# ============================================================
#             10. 最新帧摄像头
#
# 永远只保留最新帧，避免 IP 摄像头缓存。
# ============================================================

class LatestFrameCamera:

    def __init__(self, url):

        self.cap = cv2.VideoCapture(url)

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        self.lock = threading.Lock()

        self.frame = None

        self.frame_id = 0

        self.running = True

        self.thread = threading.Thread(
            target=self._reader,
            daemon=True
        )

        self.thread.start()


    def _reader(self):

        while self.running:

            ret, frame = self.cap.read()

            if ret:

                with self.lock:

                    self.frame = frame

                    self.frame_id += 1

            else:

                time.sleep(0.001)


    def read(self):

        with self.lock:

            if self.frame is None:

                return None, self.frame_id

            return (
                self.frame.copy(),
                self.frame_id
            )


    def stop(self):

        self.running = False

        self.thread.join(
            timeout=1
        )

        self.cap.release()


# ============================================================
#                    工具函数
# ============================================================

def clamp(
    value,
    minimum,
    maximum
):

    return max(
        minimum,
        min(
            value,
            maximum
        )
    )


# ============================================================
#                    图像缩放
# ============================================================

def resize_frame(frame):

    height, width = frame.shape[:2]

    if width <= PROCESS_WIDTH:

        return frame

    scale = (
        PROCESS_WIDTH / width
    )

    new_width = PROCESS_WIDTH

    new_height = int(
        height * scale
    )

    return cv2.resize(
        frame,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
#               创建管道绿色 Mask
#
# 只有管道使用 HSV。
# ============================================================

def create_pipe_mask(frame):

    roi = frame[
        PIPE_Y_TOP:PIPE_Y_BOTTOM,
        PIPE_X1:PIPE_X2
    ]

    if roi.size == 0:

        return None


    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )


    lower = np.array(
        [
            PIPE_H_LOW,
            PIPE_S_LOW,
            PIPE_V_LOW
        ],
        dtype=np.uint8
    )

    upper = np.array(
        [
            PIPE_H_HIGH,
            PIPE_S_HIGH,
            PIPE_V_HIGH
        ],
        dtype=np.uint8
    )


    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )


    # --------------------------------------------------------
    # 对管道背景降噪
    # --------------------------------------------------------

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    return mask


# ============================================================
#              计算轮廓圆形度
# ============================================================

def get_circularity(contour):

    area = cv2.contourArea(
        contour
    )

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter <= 0:

        return 0.0

    return (
        4.0
        * np.pi
        * area
        /
        (
            perimeter
            *
            perimeter
        )
    )


# ============================================================
#              创建“非管道”区域
#
# 思路：
#
#     绿色 = 管道
#
# 所以：
#
#     非绿色 = 候选钢球 / 噪声
#
# 然后通过面积、圆形度、位置筛选。
# ============================================================

def create_non_pipe_mask(frame):

    roi = frame[
        PIPE_Y_TOP:PIPE_Y_BOTTOM,
        PIPE_X1:PIPE_X2
    ]

    if roi.size == 0:

        return None


    pipe_mask = create_pipe_mask(
        frame
    )


    if pipe_mask is None:

        return None


    # --------------------------------------------------------
    # 非绿色部分
    # --------------------------------------------------------

    non_pipe = cv2.bitwise_not(
        pipe_mask
    )


    # --------------------------------------------------------
    # 钢球所在区域
    #
    # 只处理管道中间部分，减少白纸、桌面的干扰。
    # --------------------------------------------------------

    h, w = non_pipe.shape

    zone_y1 = int(
        h * BALL_ZONE_TOP_RATIO
    )

    zone_y2 = int(
        h * BALL_ZONE_BOTTOM_RATIO
    )


    zone = np.zeros_like(
        non_pipe
    )


    zone[
        zone_y1:zone_y2,
        :
    ] = 255


    non_pipe = cv2.bitwise_and(
        non_pipe,
        zone
    )


    # --------------------------------------------------------
    # 去掉非常小的噪声
    # --------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    non_pipe = cv2.morphologyEx(
        non_pipe,
        cv2.MORPH_OPEN,
        kernel
    )

    non_pipe = cv2.morphologyEx(
        non_pipe,
        cv2.MORPH_CLOSE,
        kernel
    )


    return non_pipe


# ============================================================
#             找候选钢球
#
# 注意：
#
# 完全不看钢球 HSV。
#
# ============================================================

def find_ball_candidates(
    frame,
    search_box
):

    x1, y1, x2, y2 = search_box

    frame_h, frame_w = frame.shape[:2]


    x1 = clamp(
        x1,
        0,
        frame_w - 1
    )

    y1 = clamp(
        y1,
        0,
        frame_h - 1
    )

    x2 = clamp(
        x2,
        1,
        frame_w
    )

    y2 = clamp(
        y2,
        1,
        frame_h
    )


    if x2 <= x1 or y2 <= y1:

        return []


    # --------------------------------------------------------
    # 整个管道区域的非管道 Mask
    # --------------------------------------------------------

    non_pipe = create_non_pipe_mask(
        frame
    )


    if non_pipe is None:

        return []


    # --------------------------------------------------------
    # 截取搜索范围
    # --------------------------------------------------------

    search = non_pipe[
        y1:y2,
        x1:x2
    ]


    if search.size == 0:

        return []


    # --------------------------------------------------------
    # 找轮廓
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        search,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    candidates = []


    for contour in contours:

        area = cv2.contourArea(
            contour
        )


        # ====================================================
        # 1. 面积
        # ====================================================

        if area < MIN_BALL_AREA:

            continue


        if area > MAX_BALL_AREA:

            continue


        # ====================================================
        # 2. 圆形度
        # ====================================================

        circularity = get_circularity(
            contour
        )


        if circularity < MIN_CIRCULARITY:

            continue


        # ====================================================
        # 3. 外接矩形
        # ====================================================

        rx, ry, rw, rh = cv2.boundingRect(
            contour
        )


        if rw <= 0 or rh <= 0:

            continue


        aspect_ratio = (
            rw / rh
        )


        if (
            aspect_ratio
            <
            MIN_ASPECT_RATIO
        ):

            continue


        if (
            aspect_ratio
            >
            MAX_ASPECT_RATIO
        ):

            continue


        # ====================================================
        # 4. 最小外接圆
        # ====================================================

        (
            cx,
            cy
        ), radius = cv2.minEnclosingCircle(
            contour
        )


        if radius < MIN_BALL_RADIUS:

            continue


        if radius > MAX_BALL_RADIUS:

            continue


        # ====================================================
        # 5. 重心
        # ====================================================

        M = cv2.moments(
            contour
        )


        if M["m00"] == 0:

            continue


        center_x = (
            M["m10"]
            /
            M["m00"]
        )


        center_y = (
            M["m01"]
            /
            M["m00"]
        )


        # ----------------------------------------------------
        # 转回整张图的坐标
        # ----------------------------------------------------

        global_x = (
            center_x
            +
            x1
        )

        global_y = (
            center_y
            +
            y1
        )


        # ====================================================
        # 6. 候选评分
        #
        # 这里最重要的是：
        #
        # 圆形
        # +
        # 尺寸
        # +
        # 位置
        #
        # ====================================================

        circle_score = min(
            circularity,
            1.0
        )


        # ----------------------------------------------------
        # 理想钢球圆形度
        # ----------------------------------------------------

        if circle_score >= 0.80:

            shape_score = 1.0

        elif circle_score >= 0.60:

            shape_score = 0.8

        else:

            shape_score = 0.5


        # ----------------------------------------------------
        # 半径评分
        # ----------------------------------------------------

        radius_center = (
            MIN_BALL_RADIUS
            +
            MAX_BALL_RADIUS
        ) / 2.0


        radius_range = (
            MAX_BALL_RADIUS
            -
            MIN_BALL_RADIUS
        ) / 2.0


        radius_score = max(
            0.0,
            1.0
            -
            abs(
                radius
                -
                radius_center
            )
            /
            max(
                radius_range,
                1
            )
        )


        score = (

            shape_score
            *
            0.55

            +

            radius_score
            *
            0.45

        )


        candidates.append(
            {
                "x": global_x,
                "y": global_y,
                "radius": radius,
                "area": area,
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
                "score": score
            }
        )


    # ========================================================
    # 排序
    # ========================================================

    candidates.sort(
        key=lambda item:
        item["score"],
        reverse=True
    )


    return candidates


# ============================================================
#                 选择正确的钢球
#
# 如果已经锁定：
#
#     离预测位置近
#     +
#     大小相似
#     +
#     圆形度好
#
# 才接受。
# ============================================================

def choose_candidate(
    candidates,
    previous_ball=None,
    predicted_position=None,
    strict=False
):

    if not candidates:

        return None


    # ========================================================
    # 第一次寻找
    # ========================================================

    if previous_ball is None:

        return candidates[0]


    best = None

    best_score = -999.0


    # ========================================================
    # 已经锁定
    # ========================================================

    for candidate in candidates:

        cx = candidate["x"]
        cy = candidate["y"]


        # ----------------------------------------------------
        # 距离上一位置
        # ----------------------------------------------------

        dx = (
            cx
            -
            previous_ball["x"]
        )

        dy = (
            cy
            -
            previous_ball["y"]
        )


        distance = math.sqrt(
            dx * dx
            +
            dy * dy
        )


        # ----------------------------------------------------
        # 预测位置
        # ----------------------------------------------------

        if predicted_position is not None:

            pdx = (
                cx
                -
                predicted_position[0]
            )

            pdy = (
                cy
                -
                predicted_position[1]
            )


            predicted_distance = math.sqrt(
                pdx * pdx
                +
                pdy * pdy
            )

        else:

            predicted_distance = distance


        # ----------------------------------------------------
        # 附近候选优先
        # ----------------------------------------------------

        max_distance = (

            55
            if strict
            else 100

        )


        if predicted_distance > max_distance:

            continue


        distance_score = max(
            0.0,
            1.0
            -
            predicted_distance
            /
            max_distance
        )


        # ----------------------------------------------------
        # 半径一致性
        # ----------------------------------------------------

        old_radius = previous_ball[
            "radius"
        ]


        radius_ratio = (

            candidate["radius"]
            /
            max(
                old_radius,
                1
            )

        )


        if strict:

            radius_ok = (
                0.70
                <=
                radius_ratio
                <=
                1.30
            )

        else:

            radius_ok = (
                0.55
                <=
                radius_ratio
                <=
                1.50
            )


        if not radius_ok:

            continue


        radius_score = max(
            0.0,
            1.0
            -
            abs(
                1.0
                -
                radius_ratio
            )
        )


        # ----------------------------------------------------
        # 最终评分
        # ----------------------------------------------------

        total_score = (

            candidate["score"]
            *
            0.45

            +

            distance_score
            *
            0.40

            +

            radius_score
            *
            0.15

        )


        if total_score > best_score:

            best_score = total_score

            best = candidate


    return best


# ============================================================
#                    像素 → cm
# ============================================================

def pixel_to_position_cm(
    x
):

    center_x = (

        PIPE_X1
        +
        PIPE_X2

    ) / 2.0


    pipe_pixel_length = (

        PIPE_X2
        -
        PIPE_X1

    )


    if pipe_pixel_length <= 0:

        return 0.0


    return (

        (
            x
            -
            center_x
        )
        /
        pipe_pixel_length
        *
        PIPE_LENGTH_CM

    )


# ============================================================
#                    像素速度 → cm/s
# ============================================================

def calculate_speed(
    current_x,
    previous_x,
    dt
):

    if dt <= 0:

        return 0.0


    dx_pixel = (
        current_x
        -
        previous_x
    )


    pipe_pixel_length = (

        PIPE_X2
        -
        PIPE_X1

    )


    if pipe_pixel_length <= 0:

        return 0.0


    dx_cm = (

        dx_pixel
        /
        pipe_pixel_length
        *
        PIPE_LENGTH_CM

    )


    return (
        dx_cm
        /
        dt
    )


# ============================================================
#                     主程序
# ============================================================

def main():

    print("=" * 60)

    print(
        "钢球实时检测 / 锁定系统"
    )

    print("=" * 60)

    print(
        f"手机: {VIDEO_URL}"
    )

    print(
        f"管道长度: {PIPE_LENGTH_CM} cm"
    )

    print(
        f"管道 X: {PIPE_X1} ~ {PIPE_X2}"
    )

    print(
        f"管道 Y: {PIPE_Y_TOP} ~ {PIPE_Y_BOTTOM}"
    )

    print()

    print(
        "Q = 退出"
    )

    print(
        "R = 重新锁定"
    )

    print()


    # ========================================================
    # 摄像头
    # ========================================================

    camera = LatestFrameCamera(
        VIDEO_URL
    )


    time.sleep(
        1.5
    )


    if not camera.cap.isOpened():

        print(
            "手机摄像头连接失败！"
        )

        camera.stop()

        return


    print(
        "摄像头连接成功！"
    )


    # ========================================================
    # 钢球状态
    # ========================================================

    ball = None

    previous_x = None

    previous_time = None

    velocity_cm_s = 0.0

    lost_frames = 0


    # ========================================================
    # 自检状态
    # ========================================================

    verify_mode = False

    verify_count = 0


    # ========================================================
    # FPS
    # ========================================================

    process_fps = 0.0

    process_count = 0

    process_start = time.perf_counter()


    camera_fps = 0.0

    camera_count = 0

    last_frame_id = -1

    camera_start = time.perf_counter()


    try:

        while True:

            # =================================================
            # 读取最新帧
            # =================================================

            frame, frame_id = camera.read()


            if frame is None:

                continue


            # -------------------------------------------------
            # 缩放
            # -------------------------------------------------

            frame = resize_frame(
                frame
            )


            now = time.perf_counter()


            # =================================================
            # PROCESS FPS
            # =================================================

            process_count += 1


            if (
                now
                -
                process_start
                >=
                1.0
            ):

                process_fps = (

                    process_count
                    /
                    (
                        now
                        -
                        process_start
                    )

                )

                process_count = 0

                process_start = now


            # =================================================
            # CAMERA FPS
            # =================================================

            if last_frame_id < 0:

                last_frame_id = frame_id

            elif frame_id != last_frame_id:

                camera_count += (

                    frame_id
                    -
                    last_frame_id

                )

                last_frame_id = frame_id


            if (
                now
                -
                camera_start
                >=
                1.0
            ):

                camera_fps = (

                    camera_count
                    /
                    (
                        now
                        -
                        camera_start
                    )

                )

                camera_count = 0

                camera_start = now


            # =================================================
            #                计算搜索区域
            # =================================================

            if ball is None:

                # ---------------------------------------------
                # 第一次搜索整条管道
                # ---------------------------------------------

                search_box = (
                    PIPE_X1,
                    int(
                        PIPE_Y_TOP
                        +
                        (
                            PIPE_Y_BOTTOM
                            -
                            PIPE_Y_TOP
                        )
                        *
                        BALL_ZONE_TOP_RATIO
                    ),
                    PIPE_X2,
                    int(
                        PIPE_Y_TOP
                        +
                        (
                            PIPE_Y_BOTTOM
                            -
                            PIPE_Y_TOP
                        )
                        *
                        BALL_ZONE_BOTTOM_RATIO
                    )
                )


                predicted = None

            else:

                # ---------------------------------------------
                # 根据速度预测
                # ---------------------------------------------

                if previous_time is not None:

                    dt_predict = (

                        now
                        -
                        previous_time

                    )

                else:

                    dt_predict = 0


                pixel_speed = (

                    velocity_cm_s

                    *
                    (
                        PIPE_X2
                        -
                        PIPE_X1
                    )

                    /
                    PIPE_LENGTH_CM

                )


                predicted_x = (

                    ball["x"]
                    +
                    pixel_speed
                    *
                    dt_predict

                )


                predicted_y = (
                    ball["y"]
                )


                predicted = (
                    predicted_x,
                    predicted_y
                )


                # ---------------------------------------------
                # 正常搜索
                # ---------------------------------------------

                if lost_frames == 0:

                    search_width = (
                        TRACK_SEARCH_WIDTH
                    )

                    search_height = (
                        TRACK_SEARCH_HEIGHT
                    )

                else:

                    search_width = (
                        LOST_SEARCH_WIDTH
                    )

                    search_height = (
                        LOST_SEARCH_HEIGHT
                    )


                search_box = (

                    int(
                        predicted_x
                        -
                        search_width / 2
                    ),

                    int(
                        predicted_y
                        -
                        search_height / 2
                    ),

                    int(
                        predicted_x
                        +
                        search_width / 2
                    ),

                    int(
                        predicted_y
                        +
                        search_height / 2
                    )

                )


            # =================================================
            # 查找候选
            # =================================================

            candidates = find_ball_candidates(

                frame,

                search_box

            )


            # =================================================
            # 选择候选
            # =================================================

            candidate = choose_candidate(

                candidates,

                previous_ball=ball,

                predicted_position=predicted,

                strict=verify_mode

            )


            # =================================================
            # 自检模式
            # =================================================

            if verify_mode:

                if candidate is None:

                    verify_count += 1

                    if (
                        verify_count
                        >=
                        VERIFY_FRAMES
                    ):

                        print(
                            "[SELF-CHECK] "
                            "候选不可信，"
                            "保持原钢球位置。"
                        )

                        verify_mode = False

                        verify_count = 0

                else:

                    # -----------------------------------------
                    # 检查跳跃距离
                    # -----------------------------------------

                    dx = (
                        candidate["x"]
                        -
                        ball["x"]
                    )

                    dy = (
                        candidate["y"]
                        -
                        ball["y"]
                    )


                    distance = math.sqrt(
                        dx * dx
                        +
                        dy * dy
                    )


                    # -----------------------------------------
                    # 检查半径
                    # -----------------------------------------

                    radius_ratio = (

                        candidate["radius"]
                        /
                        max(
                            ball["radius"],
                            1
                        )

                    )


                    radius_ok = (

                        0.70
                        <=
                        radius_ratio
                        <=
                        1.30

                    )


                    if (

                        distance <= 55

                        and

                        radius_ok

                    ):

                        verify_count += 1


                        if (
                            verify_count
                            >=
                            VERIFY_FRAMES
                        ):

                            print(
                                "[SELF-CHECK] "
                                "候选通过，"
                                "恢复锁定。"
                            )


                            ball = (
                                candidate
                            )


                            previous_x = (
                                candidate["x"]
                            )


                            previous_time = (
                                now
                            )


                            lost_frames = 0


                            verify_mode = False

                            verify_count = 0


                    else:

                        verify_count = 0


            # =================================================
            # 正常跟踪
            # =================================================

            else:

                if candidate is not None:

                    # =========================================
                    # 第一次锁定
                    # =========================================

                    if ball is None:

                        ball = candidate

                        previous_x = (
                            candidate["x"]
                        )

                        previous_time = now

                        velocity_cm_s = 0.0

                        lost_frames = 0


                        print(
                            "[LOCK] "
                            "钢球已经锁定"
                        )


                    # =========================================
                    # 已经锁定
                    # =========================================

                    else:

                        dx_pixel = (

                            candidate["x"]
                            -
                            previous_x

                        )


                        dt = (

                            now
                            -
                            previous_time

                        )


                        raw_speed = calculate_speed(

                            candidate["x"],

                            previous_x,

                            dt

                        )


                        # -------------------------------------
                        # 单帧位置变化 cm
                        # -------------------------------------

                        jump_cm = abs(

                            dx_pixel
                            /
                            max(
                                PIPE_X2
                                -
                                PIPE_X1,
                                1
                            )
                            *
                            PIPE_LENGTH_CM

                        )


                        # -------------------------------------
                        # 判断异常
                        # -------------------------------------

                        suspicious = (

                            jump_cm
                            >
                            MAX_SINGLE_JUMP_CM

                            or

                            abs(raw_speed)
                            >
                            MAX_REASONABLE_SPEED_CM_S

                        )


                        # =====================================
                        # 正常
                        # =====================================

                        if not suspicious:

                            velocity_cm_s = (

                                (
                                    1
                                    -
                                    VELOCITY_ALPHA
                                )
                                *
                                velocity_cm_s

                                +

                                VELOCITY_ALPHA
                                *
                                raw_speed

                            )


                            ball = (
                                candidate
                            )


                            previous_x = (
                                candidate["x"]
                            )


                            previous_time = (
                                now
                            )


                            lost_frames = 0


                        # =====================================
                        # 异常跳点
                        # =====================================

                        else:

                            print()
                            print(
                                "[SELF-CHECK] "
                                "发现疑似误识别！"
                            )

                            print(
                                f"原位置: "
                                f"{previous_x:.1f}px"
                            )

                            print(
                                f"新候选: "
                                f"{candidate['x']:.1f}px"
                            )

                            print(
                                f"单帧跳动: "
                                f"{jump_cm:.2f} cm"
                            )

                            print(
                                f"瞬时速度: "
                                f"{raw_speed:.2f} cm/s"
                            )

                            print(
                                "暂时不接受该候选，"
                                "开始严格验证。"
                            )


                            verify_mode = True

                            verify_count = 0


                # =================================================
                # 没有候选
                # =================================================

                else:

                    lost_frames += 1


                    # ---------------------------------------------
                    # 短时间没找到
                    #
                    # 保持原来的锁定结果
                    # ---------------------------------------------

                    if (
                        lost_frames
                        <=
                        MAX_LOST_FRAMES
                    ):

                        pass


                    # ---------------------------------------------
                    # 长时间没有找到
                    #
                    # 重新搜索
                    # ---------------------------------------------

                    else:

                        print(
                            "[LOCK] "
                            "钢球长时间丢失，"
                            "重新搜索。"
                        )


                        ball = None

                        previous_x = None

                        previous_time = None

                        velocity_cm_s = 0.0

                        lost_frames = 0


            # =================================================
            #               显示钢球
            # =================================================

            if ball is not None:

                bx = int(
                    ball["x"]
                )

                by = int(
                    ball["y"]
                )

                radius = max(
                    int(
                        ball["radius"]
                    ),
                    6
                )


                # ------------------------------------------------
                # 钢球红圈
                # ------------------------------------------------

                cv2.circle(

                    frame,

                    (
                        bx,
                        by
                    ),

                    radius,

                    (0, 0, 255),

                    2

                )


                # ------------------------------------------------
                # 钢球中心
                # ------------------------------------------------

                cv2.circle(

                    frame,

                    (
                        bx,
                        by
                    ),

                    4,

                    (0, 0, 255),

                    -1

                )


                # =================================================
                # Position
                # =================================================

                position_cm = (
                    pixel_to_position_cm(
                        ball["x"]
                    )
                )


                distance_cm = abs(
                    position_cm
                )


                # =================================================
                # 状态
                # =================================================

                if verify_mode:

                    state = "VERIFY"

                    state_color = (
                        0,
                        255,
                        255
                    )

                elif lost_frames > 0:

                    state = "TRACKING"

                    state_color = (
                        0,
                        255,
                        255
                    )

                else:

                    state = "LOCKED"

                    state_color = (
                        0,
                        255,
                        0
                    )


                # ------------------------------------------------
                # 状态
                # ------------------------------------------------

                cv2.putText(

                    frame,

                    f"BALL: {state}",

                    (
                        20,
                        35
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    state_color,

                    2

                )


                # ------------------------------------------------
                # Position
                # ------------------------------------------------

                cv2.putText(

                    frame,

                    (
                        f"Position: "
                        f"{position_cm:+.2f} cm"
                    ),

                    (
                        20,
                        68
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (255, 255, 255),

                    2

                )


                # ------------------------------------------------
                # 距离中心
                # ------------------------------------------------

                cv2.putText(

                    frame,

                    (
                        f"Distance: "
                        f"{distance_cm:.2f} cm"
                    ),

                    (
                        20,
                        100
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (255, 255, 255),

                    2

                )


            else:

                cv2.putText(

                    frame,

                    "SEARCHING BALL...",

                    (
                        20,
                        35
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0, 0, 255),

                    2

                )


            # =================================================
            # 三条水平平行线
            # =================================================

            cv2.line(

                frame,

                (
                    PIPE_X1,
                    PIPE_Y_TOP
                ),

                (
                    PIPE_X2,
                    PIPE_Y_TOP
                ),

                (0, 255, 0),

                2

            )


            cv2.line(

                frame,

                (
                    PIPE_X1,
                    PIPE_Y_CENTER
                ),

                (
                    PIPE_X2,
                    PIPE_Y_CENTER
                ),

                (255, 0, 0),

                2

            )


            cv2.line(

                frame,

                (
                    PIPE_X1,
                    PIPE_Y_BOTTOM
                ),

                (
                    PIPE_X2,
                    PIPE_Y_BOTTOM
                ),

                (0, 255, 0),

                2

            )


            # =================================================
            # FPS
            # =================================================

            cv2.putText(

                frame,

                f"PROC: {process_fps:.1f} FPS",

                (
                    frame.shape[1] - 205,
                    32
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.60,

                (255, 255, 255),

                2

            )


            cv2.putText(

                frame,

                f"CAM: {camera_fps:.1f} FPS",

                (
                    frame.shape[1] - 205,
                    62
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.60,

                (0, 255, 0),

                2

            )


            # =================================================
            # 唯一显示窗口
            # =================================================

            cv2.imshow(

                "Steel Ball Tracking",

                frame

            )


            # =================================================
            # 键盘
            # =================================================

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )


            if key == ord("q"):

                break


            elif key == ord("r"):

                print(
                    "[USER] "
                    "重新搜索钢球"
                )


                ball = None

                previous_x = None

                previous_time = None

                velocity_cm_s = 0.0

                lost_frames = 0

                verify_mode = False

                verify_count = 0


    finally:

        camera.stop()

        cv2.destroyAllWindows()


    print(
        "程序结束"
    )


# ============================================================
#                    程序入口
# ============================================================

if __name__ == "__main__":

    main()