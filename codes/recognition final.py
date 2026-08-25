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
#                    2. 管道实际长度
# ============================================================

PIPE_LENGTH_CM = 25.0


# ============================================================
#                    3. 图像处理
# ============================================================

PROCESS_WIDTH = 1280


# ============================================================
#              4. 管道 HSV
#
# 只用于识别绿色管道
# 不用于判断钢球颜色
# ============================================================

PIPE_H_LOW = 50
PIPE_H_HIGH = 90

PIPE_S_LOW = 40
PIPE_S_HIGH = 180

PIPE_V_LOW = 25
PIPE_V_HIGH = 200


# ============================================================
#              5. 管道位置自动检测参数
# ============================================================

MIN_PIPE_AREA = 3000


# ============================================================
#              6. 钢球大小约束
#
# 不判断钢球 HSV
#
# 只利用：
#     面积
#     圆形度
#     半径
#     宽高比
# ============================================================

MIN_BALL_AREA = 20
MAX_BALL_AREA = 1800

MIN_BALL_RADIUS = 4
MAX_BALL_RADIUS = 35

MIN_CIRCULARITY = 0.30

MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 2.20


# ============================================================
#              7. 钢球在管道里面的范围
# ============================================================

BALL_VERTICAL_RATIO_TOP = 0.15
BALL_VERTICAL_RATIO_BOTTOM = 0.85


# ============================================================
#              8. 锁定参数
# ============================================================

TRACK_SEARCH_WIDTH = 180
TRACK_SEARCH_HEIGHT = 110

LOST_SEARCH_WIDTH = 300
LOST_SEARCH_HEIGHT = 150

MAX_LOST_FRAMES = 8


# ============================================================
#              9. 防止钢球跳点
#
# 注意：
#
# 这里不再使用“速度过大”判断误判。
#
# 改成：
#
#     单帧位置跳跃过大
#     OR
#     加速度过大
#
# ============================================================

MAX_SINGLE_JUMP_CM = 2.0

# 最大允许加速度
#
# 单位：cm/s²
#
# 这个值是“防止视觉误判”的阈值，
# 不是说钢球物理上绝对不可能超过这个加速度。
#
# 如果后面发现真实钢球运动时经常触发自检，
# 再适当提高这个值。
#
MAX_REASONABLE_ACCELERATION_CM_S2 = 1200.0


# ============================================================
#              10. 速度滤波
#
# 速度仍然需要计算：
#
#     1. 预测下一帧位置
#     2. 计算加速度
#
# 但不显示在画面中。
# ============================================================

VELOCITY_ALPHA = 0.20


# ============================================================
#              11. 摄像头最新帧
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
            timeout=1.0
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
#                    缩放画面
# ============================================================

def resize_frame(frame):

    h, w = frame.shape[:2]

    if w <= PROCESS_WIDTH:

        return frame

    scale = (
        PROCESS_WIDTH
        /
        w
    )

    new_w = PROCESS_WIDTH

    new_h = int(
        h * scale
    )

    return cv2.resize(
        frame,
        (
            new_w,
            new_h
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
#                创建管道绿色 Mask
# ============================================================

def create_pipe_mask(frame):

    hsv = cv2.cvtColor(
        frame,
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
    # 横向连接管道
    # --------------------------------------------------------

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            31,
            7
        )
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close
    )


    # --------------------------------------------------------
    # 去小噪声
    # --------------------------------------------------------

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            5,
            5
        )
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_open
    )


    return mask


# ============================================================
#              自动寻找绿色管道
# ============================================================

def find_pipe(frame):

    mask = create_pipe_mask(
        frame
    )


    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if not contours:

        return None


    candidates = []


    for contour in contours:

        area = cv2.contourArea(
            contour
        )


        if area < MIN_PIPE_AREA:

            continue


        x, y, w, h = cv2.boundingRect(
            contour
        )


        if w < 300:

            continue


        if h < 20:

            continue


        aspect = (
            w /
            max(
                h,
                1
            )
        )


        if aspect < 3.0:

            continue


        candidates.append(
            (
                area,
                x,
                y,
                w,
                h
            )
        )


    if not candidates:

        return None


    candidates.sort(
        reverse=True
    )


    _, x, y, w, h = candidates[0]


    # --------------------------------------------------------
    # 给管道留一点余量
    # --------------------------------------------------------

    padding_x = 5
    padding_y = 5


    x1 = max(
        0,
        x - padding_x
    )


    y1 = max(
        0,
        y - padding_y
    )


    x2 = min(
        frame.shape[1],
        x + w + padding_x
    )


    y2 = min(
        frame.shape[0],
        y + h + padding_y
    )


    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
#              计算钢球圆形度
# ============================================================

def circularity(contour):

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
#             获取钢球搜索区域
# ============================================================

def get_ball_zone(
    pipe_box
):

    x1, y1, x2, y2 = pipe_box


    pipe_height = (
        y2 - y1
    )


    zone_top = int(
        y1
        +
        pipe_height
        *
        BALL_VERTICAL_RATIO_TOP
    )


    zone_bottom = int(
        y1
        +
        pipe_height
        *
        BALL_VERTICAL_RATIO_BOTTOM
    )


    return (
        x1,
        zone_top,
        x2,
        zone_bottom
    )


# ============================================================
#              创建管道内部非绿色区域
# ============================================================

def create_non_green_inside_pipe(
    frame,
    pipe_box
):

    x1, y1, x2, y2 = get_ball_zone(
        pipe_box
    )


    roi = frame[
        y1:y2,
        x1:x2
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


    pipe_mask = cv2.inRange(
        hsv,
        lower,
        upper
    )


    # --------------------------------------------------------
    # 绿色 = 管道
    # 非绿色 = 候选钢球/其他物体
    # --------------------------------------------------------

    non_green = cv2.bitwise_not(
        pipe_mask
    )


    # --------------------------------------------------------
    # 轻量去噪
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            3,
            3
        )
    )


    non_green = cv2.morphologyEx(
        non_green,
        cv2.MORPH_OPEN,
        kernel
    )


    return (
        non_green,
        x1,
        y1
    )


# ============================================================
#             判断钢球周围是不是绿色
# ============================================================

def green_surround_score(
    hsv,
    x,
    y,
    radius
):

    h, w = hsv.shape[:2]


    outer = max(
        int(radius * 2.5),
        10
    )


    inner = max(
        int(radius * 1.3),
        5
    )


    x1 = max(
        0,
        int(x - outer)
    )


    y1 = max(
        0,
        int(y - outer)
    )


    x2 = min(
        w,
        int(x + outer + 1)
    )


    y2 = min(
        h,
        int(y + outer + 1)
    )


    if x2 <= x1 or y2 <= y1:

        return 0.0


    local = hsv[
        y1:y2,
        x1:x2
    ]


    yy, xx = np.ogrid[
        y1:y2,
        x1:x2
    ]


    distance = np.sqrt(
        (xx - x) ** 2
        +
        (yy - y) ** 2
    )


    ring = (
        (distance >= inner)
        &
        (distance <= outer)
    )


    if not np.any(ring):

        return 0.0


    H = local[:, :, 0]
    S = local[:, :, 1]
    V = local[:, :, 2]


    green = (
        (H >= PIPE_H_LOW)
        &
        (H <= PIPE_H_HIGH)
        &
        (S >= PIPE_S_LOW)
        &
        (V >= PIPE_V_LOW)
    )


    return float(
        green[ring].mean()
    )


# ============================================================
#              在管道内部寻找钢球
# ============================================================

def find_ball_candidates(
    frame,
    pipe_box
):

    result = create_non_green_inside_pipe(
        frame,
        pipe_box
    )


    if result is None:

        return []


    non_green, offset_x, offset_y = (
        result
    )


    zone_x1, zone_y1, zone_x2, zone_y2 = (
        get_ball_zone(
            pipe_box
        )
    )


    hsv_roi = cv2.cvtColor(
        frame[
            zone_y1:zone_y2,
            zone_x1:zone_x2
        ],
        cv2.COLOR_BGR2HSV
    )


    contours, _ = cv2.findContours(
        non_green,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    candidates = []


    for contour in contours:

        # ----------------------------------------------------
        # 面积
        # ----------------------------------------------------

        area = cv2.contourArea(
            contour
        )


        if area < MIN_BALL_AREA:

            continue


        if area > MAX_BALL_AREA:

            continue


        # ----------------------------------------------------
        # 圆形度
        # ----------------------------------------------------

        shape = circularity(
            contour
        )


        if shape < MIN_CIRCULARITY:

            continue


        # ----------------------------------------------------
        # 外接矩形
        # ----------------------------------------------------

        bx, by, bw, bh = cv2.boundingRect(
            contour
        )


        if bw <= 0 or bh <= 0:

            continue


        aspect = (
            bw /
            bh
        )


        if aspect < MIN_ASPECT_RATIO:

            continue


        if aspect > MAX_ASPECT_RATIO:

            continue


        # ----------------------------------------------------
        # 最小外接圆
        # ----------------------------------------------------

        (
            _,
            _
        ), radius = cv2.minEnclosingCircle(
            contour
        )


        if radius < MIN_BALL_RADIUS:

            continue


        if radius > MAX_BALL_RADIUS:

            continue


        # ----------------------------------------------------
        # 重心
        # ----------------------------------------------------

        M = cv2.moments(
            contour
        )


        if M["m00"] == 0:

            continue


        cx = (
            M["m10"]
            /
            M["m00"]
        )


        cy = (
            M["m01"]
            /
            M["m00"]
        )


        global_x = (
            cx
            +
            offset_x
        )


        global_y = (
            cy
            +
            offset_y
        )


        # ----------------------------------------------------
        # 周围绿色
        # ----------------------------------------------------

        ring_score = green_surround_score(

            hsv_roi,

            cx,

            cy,

            radius

        )


        if ring_score < 0.20:

            continue


        # ----------------------------------------------------
        # 综合评分
        # ----------------------------------------------------

        shape_score = min(
            shape,
            1.0
        )


        if shape_score >= 0.75:

            shape_bonus = 1.0

        elif shape_score >= 0.55:

            shape_bonus = 0.8

        else:

            shape_bonus = 0.5


        ring_bonus = min(
            ring_score,
            1.0
        )


        score = (

            shape_bonus
            *
            0.50

            +

            ring_bonus
            *
            0.50

        )


        candidates.append(
            {
                "x": global_x,
                "y": global_y,
                "radius": radius,
                "area": area,
                "circularity": shape,
                "aspect": aspect,
                "ring_score": ring_score,
                "score": score
            }
        )


    candidates.sort(
        key=lambda item:
        item["score"],
        reverse=True
    )


    return candidates


# ============================================================
#              根据上一帧选择钢球
# ============================================================

def choose_ball(
    candidates,
    previous_ball,
    predicted_position,
    strict=False
):

    if not candidates:

        return None


    # ========================================================
    # 第一次搜索
    # ========================================================

    if previous_ball is None:

        return candidates[0]


    best = None

    best_score = -999.0


    for candidate in candidates:

        # ----------------------------------------------------
        # 距离预测位置
        # ----------------------------------------------------

        dx = (
            candidate["x"]
            -
            predicted_position[0]
        )


        dy = (
            candidate["y"]
            -
            predicted_position[1]
        )


        distance = math.sqrt(
            dx * dx
            +
            dy * dy
        )


        max_distance = (

            60
            if strict
            else 100

        )


        if distance > max_distance:

            continue


        distance_score = max(
            0.0,
            1.0
            -
            distance
            /
            max_distance
        )


        # ----------------------------------------------------
        # 半径稳定
        # ----------------------------------------------------

        old_radius = (
            previous_ball["radius"]
        )


        radius_ratio = (

            candidate["radius"]
            /
            max(
                old_radius,
                1
            )

        )


        if strict:

            if not (
                0.70
                <=
                radius_ratio
                <=
                1.30
            ):

                continue

        else:

            if not (
                0.55
                <=
                radius_ratio
                <=
                1.50
            ):

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
        # 总评分
        # ----------------------------------------------------

        total_score = (

            candidate["score"]
            *
            0.50

            +

            distance_score
            *
            0.35

            +

            radius_score
            *
            0.15

        )


        if total_score > best_score:

            best_score = (
                total_score
            )

            best = candidate


    return best


# ============================================================
#                  像素 → cm
# ============================================================

def pixel_to_position_cm(
    x,
    pipe_box
):

    x1, _, x2, _ = pipe_box


    center_x = (
        x1 + x2
    ) / 2.0


    pipe_pixel_length = (
        x2 - x1
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
#                  速度计算
# ============================================================

def calculate_speed(
    current_x,
    previous_x,
    dt,
    pipe_box
):

    if dt <= 0:

        return 0.0


    x1, _, x2, _ = pipe_box


    pipe_length_pixel = (
        x2 - x1
    )


    if pipe_length_pixel <= 0:

        return 0.0


    dx_cm = (

        (
            current_x
            -
            previous_x
        )
        /
        pipe_length_pixel
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
        "钢球固定场景视觉锁定系统"
    )

    print("=" * 60)

    print(
        f"手机: {VIDEO_URL}"
    )

    print(
        f"管道长度: {PIPE_LENGTH_CM} cm"
    )

    print()

    print(
        "Q = 退出"
    )

    print(
        "R = 重新搜索"
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
            "手机摄像头连接失败"
        )

        camera.stop()

        return


    print(
        "摄像头连接成功"
    )


    # ========================================================
    # 状态
    # ========================================================

    pipe_box = None

    ball = None

    previous_x = None

    previous_time = None

    # 当前速度
    velocity_cm_s = 0.0

    # 上一时刻速度
    previous_velocity_cm_s = 0.0

    # 当前加速度
    acceleration_cm_s2 = 0.0

    lost_frames = 0


    # ========================================================
    # 自检
    # ========================================================

    verify_mode = False

    verify_count = 0


    # ========================================================
    # PROCESS FPS
    # ========================================================

    process_fps = 0.0

    process_count = 0

    process_start = time.perf_counter()


    # ========================================================
    # CAMERA FPS
    # ========================================================

    camera_fps = 0.0

    camera_count = 0

    last_frame_id = -1

    camera_start = time.perf_counter()


    try:

        while True:

            # =================================================
            # 最新帧
            # =================================================

            frame, frame_id = (
                camera.read()
            )


            if frame is None:

                continue


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

                last_frame_id = (
                    frame_id
                )

            elif frame_id != last_frame_id:

                camera_count += (

                    frame_id
                    -
                    last_frame_id

                )

                last_frame_id = (
                    frame_id
                )


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
            # 自动找管道
            # =================================================

            if (
                pipe_box is None
                or
                ball is None
            ):

                detected_pipe = (
                    find_pipe(
                        frame
                    )
                )


                if detected_pipe is not None:

                    if pipe_box is None:

                        pipe_box = (
                            detected_pipe
                        )

                    else:

                        old = np.array(
                            pipe_box,
                            dtype=np.float32
                        )


                        new = np.array(
                            detected_pipe,
                            dtype=np.float32
                        )


                        smooth = (

                            old
                            *
                            0.75

                            +

                            new
                            *
                            0.25

                        )


                        pipe_box = tuple(

                            smooth.astype(
                                int
                            )

                        )


            # =================================================
            # 管道还没找到
            # =================================================

            if pipe_box is None:

                cv2.putText(

                    frame,

                    "SEARCHING PIPE...",

                    (
                        20,
                        35
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0, 0, 255),

                    2

                )


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


                cv2.imshow(
                    "Steel Ball Tracking",
                    frame
                )


                key = (
                    cv2.waitKey(1)
                    &
                    0xFF
                )


                if key == ord("q"):

                    break


                continue


            # =================================================
            # 预测钢球
            # =================================================

            predicted = None


            if ball is not None:

                if previous_time is not None:

                    dt_predict = (

                        now
                        -
                        previous_time

                    )

                else:

                    dt_predict = 0.0


                pipe_pixel_length = (

                    pipe_box[2]
                    -
                    pipe_box[0]

                )


                if pipe_pixel_length > 0:

                    pixel_speed = (

                        velocity_cm_s
                        *
                        pipe_pixel_length
                        /
                        PIPE_LENGTH_CM

                    )

                else:

                    pixel_speed = 0.0


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


            # =================================================
            # 搜索框
            # =================================================

            if ball is None:

                search_box = (
                    get_ball_zone(
                        pipe_box
                    )
                )

            else:

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
                        predicted[0]
                        -
                        search_width / 2
                    ),

                    int(
                        predicted[1]
                        -
                        search_height / 2
                    ),

                    int(
                        predicted[0]
                        +
                        search_width / 2
                    ),

                    int(
                        predicted[1]
                        +
                        search_height / 2
                    )

                )


            # =================================================
            # 寻找候选
            # =================================================

            candidates = find_ball_candidates(

                frame,

                pipe_box

            )


            # -------------------------------------------------
            # 只保留搜索框里的候选
            # -------------------------------------------------

            sx1, sy1, sx2, sy2 = (
                search_box
            )


            filtered_candidates = []


            for candidate in candidates:

                if (

                    sx1
                    <=
                    candidate["x"]
                    <=
                    sx2

                    and

                    sy1
                    <=
                    candidate["y"]
                    <=
                    sy2

                ):

                    filtered_candidates.append(
                        candidate
                    )


            candidates = (
                filtered_candidates
            )


            # =================================================
            # 选择目标
            # =================================================

            candidate = choose_ball(

                candidates,

                previous_ball=ball,

                predicted_position=(

                    predicted
                    if predicted is not None
                    else (
                        0,
                        0
                    )

                ),

                strict=verify_mode

            )


            # =================================================
            # 自检
            # =================================================

            if verify_mode:

                if candidate is None:

                    verify_count += 1


                    if (
                        verify_count
                        >=
                        2
                    ):

                        print(
                            "[VERIFY] "
                            "没有可信候选，"
                            "保持当前锁定。"
                        )


                        verify_mode = False

                        verify_count = 0


                else:

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

                        distance <= 60

                        and

                        radius_ok

                    ):

                        verify_count += 1


                        if (
                            verify_count
                            >=
                            2
                        ):

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


                            print(
                                "[VERIFY] "
                                "重新锁定成功"
                            )


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

                        ball = (
                            candidate
                        )


                        previous_x = (
                            candidate["x"]
                        )


                        previous_time = now


                        velocity_cm_s = 0.0

                        previous_velocity_cm_s = 0.0

                        acceleration_cm_s2 = 0.0

                        lost_frames = 0


                        print(
                            "[LOCK] "
                            "钢球锁定成功"
                        )


                    else:

                        # =====================================
                        # 计算 dt
                        # =====================================

                        dt = (
                            now
                            -
                            previous_time
                        )


                        if dt > 0:

                            # ---------------------------------
                            # 当前瞬时速度
                            # ---------------------------------

                            raw_speed = (
                                calculate_speed(

                                    candidate["x"],

                                    previous_x,

                                    dt,

                                    pipe_box

                                )
                            )


                            # ---------------------------------
                            # 平滑后的当前速度
                            # ---------------------------------

                            new_velocity = (

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


                            # ---------------------------------
                            # 当前加速度
                            #
                            # a = Δv / Δt
                            # ---------------------------------

                            current_acceleration = (

                                (
                                    new_velocity
                                    -
                                    velocity_cm_s
                                )
                                /
                                dt

                            )


                        else:

                            raw_speed = (
                                velocity_cm_s
                            )

                            new_velocity = (
                                velocity_cm_s
                            )

                            current_acceleration = 0.0


                        # =====================================
                        # 单帧位置跳跃
                        # =====================================

                        jump_cm = abs(

                            candidate["x"]
                            -
                            previous_x

                        )


                        jump_cm = (

                            jump_cm

                            /

                            max(
                                pipe_box[2]
                                -
                                pipe_box[0],
                                1
                            )

                            *

                            PIPE_LENGTH_CM

                        )


                        # =====================================
                        # 判断疑似误判
                        #
                        # 现在只看：
                        #
                        # 1. 单帧位置跳跃
                        # 2. 加速度
                        #
                        # 不再直接判断速度
                        # =====================================

                        suspicious = (

                            jump_cm
                            >
                            MAX_SINGLE_JUMP_CM

                            or

                            abs(
                                current_acceleration
                            )
                            >
                            MAX_REASONABLE_ACCELERATION_CM_S2

                        )


                        # =====================================
                        # 正常
                        # =====================================

                        if not suspicious:

                            velocity_cm_s = (
                                new_velocity
                            )


                            previous_velocity_cm_s = (
                                velocity_cm_s
                            )


                            acceleration_cm_s2 = (
                                current_acceleration
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
                        # 异常
                        # =====================================

                        else:

                            print()
                            print(
                                "[SELF-CHECK] "
                                "检测到疑似误判!"
                            )


                            print(
                                f"原位置: "
                                f"{previous_x:.1f}px"
                            )


                            print(
                                f"候选位置: "
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
                                f"加速度: "
                                f"{current_acceleration:.2f} cm/s²"
                            )


                            print(
                                "速度本身不作为误判标准。"
                            )


                            print(
                                "暂不接受该候选，"
                                "进入严格自检。"
                            )


                            verify_mode = True

                            verify_count = 0


                else:

                    lost_frames += 1


                    if (
                        lost_frames
                        >
                        MAX_LOST_FRAMES
                    ):

                        print(
                            "[LOCK] "
                            "钢球长时间丢失，"
                            "重新搜索。"
                        )


                        ball = None

                        previous_x = None

                        previous_time = None

                        velocity_cm_s = 0.0

                        previous_velocity_cm_s = 0.0

                        acceleration_cm_s2 = 0.0

                        lost_frames = 0


            # =================================================
            #                    画管道
            # =================================================

            x1, y1, x2, y2 = pipe_box


            center_y = (
                y1 + y2
            ) // 2


            # 上线
            cv2.line(

                frame,

                (
                    x1,
                    y1
                ),

                (
                    x2,
                    y1
                ),

                (0, 255, 0),

                2

            )


            # 中心线
            cv2.line(

                frame,

                (
                    x1,
                    center_y
                ),

                (
                    x2,
                    center_y
                ),

                (255, 0, 0),

                2

            )


            # 下线
            cv2.line(

                frame,

                (
                    x1,
                    y2
                ),

                (
                    x2,
                    y2
                ),

                (0, 255, 0),

                2

            )


            # =================================================
            #                    显示钢球
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


                # 红色锁定圈
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


                # 中心点
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

                position = (
                    pixel_to_position_cm(

                        ball["x"],

                        pipe_box

                    )
                )


                distance = abs(
                    position
                )


                # =================================================
                # 状态
                # =================================================

                if verify_mode:

                    state = "VERIFY"

                    color = (
                        0,
                        255,
                        255
                    )

                elif lost_frames > 0:

                    state = "TRACKING"

                    color = (
                        0,
                        255,
                        255
                    )

                else:

                    state = "LOCKED"

                    color = (
                        0,
                        255,
                        0
                    )


                cv2.putText(

                    frame,

                    f"BALL: {state}",

                    (
                        20,
                        35
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    color,

                    2

                )


                cv2.putText(

                    frame,

                    (
                        f"Position: "
                        f"{position:+.2f} cm"
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


                cv2.putText(

                    frame,

                    (
                        f"Distance: "
                        f"{distance:.2f} cm"
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
            #                     FPS
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
            # 唯一窗口
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

                previous_velocity_cm_s = 0.0

                acceleration_cm_s2 = 0.0

                lost_frames = 0

                verify_mode = False

                verify_count = 0

                pipe_box = None


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