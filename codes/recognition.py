import cv2
import numpy as np
import time
import math


# ============================================================
#                         参数设置
# ============================================================

PHONE_IP = "172.30.230.152"
PHONE_PORT = 4747

VIDEO_URL = f"http://{PHONE_IP}:{PHONE_PORT}/video"


# ============================================================
#                         水管 ROI
# ============================================================
#
# !!! 这里请使用你原来程序中已经调好的 ROI 参数 !!!
#
# x1, y1 = 左上角
# x2, y2 = 右下角
#
# ============================================================

PIPE_X1 = 100
PIPE_Y1 = 100
PIPE_X2 = 1000
PIPE_Y2 = 600


# ============================================================
#                    管道实际长度
# ============================================================

PIPE_LENGTH_CM = 25.0


# ============================================================
#                  管道实际两端 X 坐标
# ============================================================
#
# 如果你的 ROI 正好覆盖整个管道：
#
# 左端 = PIPE_X1
# 右端 = PIPE_X2
#
# 如果以后发现管道实际两端不是 ROI 边缘，
# 只修改这里即可。
#
# ============================================================

PIPE_START_X = PIPE_X1
PIPE_END_X = PIPE_X2


# ============================================================
#                         DEBUG
# ============================================================

DEBUG_MODE = True


# ============================================================
#                       钢球检测参数
# ============================================================

MIN_RADIUS = 5
MAX_RADIUS = 80

MIN_CIRCLE_DISTANCE = 15

HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 20


# ============================================================
#                       跟踪参数
# ============================================================

MAX_TRACK_DISTANCE = 120

MAX_MISSING_FRAMES = 8

MOTION_HISTORY_LENGTH = 8

MOVEMENT_THRESHOLD = 4


# ============================================================
#                       速度参数
# ============================================================

SPEED_HISTORY_LENGTH = 5


# ============================================================
#                       OpenCV 字体
# ============================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX

WINDOW_NAME = "Steel Ball Detection"


# ============================================================
#                       工具函数
# ============================================================

def clamp(value, min_value, max_value):
    """
    将 value 限制在 min_value ~ max_value 之间
    """

    return max(
        min_value,
        min(value, max_value)
    )


def point_distance(p1, p2):
    """
    计算两个点之间的二维距离
    """

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
#           计算钢球相对于管道中心的位置
# ============================================================

def pixel_to_cm(pixel_x):
    """
    计算钢球相对于管道中心的位置。

    管道总长度 = 25 cm

    左端：
        -12.5 cm

    中心：
         0 cm

    右端：
        +12.5 cm

    返回：
        带正负号的位置
    """

    pipe_pixel_length = (
        PIPE_END_X - PIPE_START_X
    )

    if pipe_pixel_length <= 0:
        return 0.0

    # 管道中心 X
    pipe_center_x = (
        PIPE_START_X + PIPE_END_X
    ) / 2.0

    # 钢球距离中心的像素距离
    pixel_offset = (
        pixel_x - pipe_center_x
    )

    # 转换为 cm
    position_cm = (
        pixel_offset
        /
        pipe_pixel_length
        *
        PIPE_LENGTH_CM
    )

    return position_cm


# ============================================================
#              计算钢球距离管道中心的距离
# ============================================================

def calculate_distance_to_center(pixel_x):
    """
    计算钢球距离管道中心的距离。

    注意：
        这里返回绝对值。

    例如：

        中心左边 5 cm -> 5 cm

        中心右边 5 cm -> 5 cm
    """

    pipe_pixel_length = (
        PIPE_END_X - PIPE_START_X
    )

    if pipe_pixel_length <= 0:
        return 0.0

    pipe_center_x = (
        PIPE_START_X + PIPE_END_X
    ) / 2.0

    pixel_distance = abs(
        pixel_x - pipe_center_x
    )

    distance_cm = (
        pixel_distance
        /
        pipe_pixel_length
        *
        PIPE_LENGTH_CM
    )

    return distance_cm


# ============================================================
#                   判断钢球在哪一侧
# ============================================================

def get_center_side(pixel_x):
    """
    返回：

    LEFT
    CENTER
    RIGHT
    """

    pipe_center_x = (
        PIPE_START_X + PIPE_END_X
    ) / 2.0

    # 设置一个很小的中心容差
    center_tolerance = 3

    if pixel_x < pipe_center_x - center_tolerance:

        return "LEFT"

    elif pixel_x > pipe_center_x + center_tolerance:

        return "RIGHT"

    else:

        return "CENTER"


# ============================================================
#                    钢球候选评分
# ============================================================

def calculate_ball_score(
    gray,
    hsv,
    x,
    y,
    r
):

    h, w = gray.shape

    x1 = max(
        0,
        int(x - r)
    )

    y1 = max(
        0,
        int(y - r)
    )

    x2 = min(
        w,
        int(x + r)
    )

    y2 = min(
        h,
        int(y + r)
    )

    if x2 <= x1 or y2 <= y1:
        return -999

    roi_gray = gray[
        y1:y2,
        x1:x2
    ]

    roi_hsv = hsv[
        y1:y2,
        x1:x2
    ]

    if roi_gray.size == 0:
        return -999

    # ========================================================
    #                       圆形 Mask
    # ========================================================

    mask = np.zeros(
        roi_gray.shape,
        dtype=np.uint8
    )

    cx = int(x - x1)
    cy = int(y - y1)

    cv2.circle(
        mask,
        (cx, cy),
        int(r * 0.75),
        255,
        -1
    )

    pixels_gray = roi_gray[
        mask > 0
    ]

    if len(pixels_gray) < 10:
        return -999

    # ========================================================
    #                       灰度均匀度
    # ========================================================

    mean_gray = float(
        np.mean(pixels_gray)
    )

    std_gray = float(
        np.std(pixels_gray)
    )

    uniformity_score = clamp(
        1.0 - std_gray / 80.0,
        0.0,
        1.0
    )

    # ========================================================
    #                       饱和度
    # ========================================================

    pixels_hsv = roi_hsv[
        mask > 0
    ]

    mean_saturation = float(
        np.mean(
            pixels_hsv[:, 1]
        )
    )

    saturation_score = clamp(
        1.0 -
        mean_saturation / 180.0,
        0.0,
        1.0
    )

    # ========================================================
    #                       边缘
    # ========================================================

    edge = cv2.Canny(
        roi_gray,
        HOUGH_PARAM1,
        HOUGH_PARAM1 * 2
    )

    circle_mask = np.zeros_like(
        mask
    )

    cv2.circle(
        circle_mask,
        (cx, cy),
        int(r * 0.95),
        255,
        2
    )

    edge_pixels = edge[
        circle_mask > 0
    ]

    if len(edge_pixels) > 0:

        edge_score = float(
            np.count_nonzero(
                edge_pixels
            )
            /
            len(edge_pixels)
        )

    else:

        edge_score = 0.0

    edge_score = clamp(
        edge_score * 2.0,
        0.0,
        1.0
    )

    # ========================================================
    #                       综合评分
    # ========================================================

    score = (
        uniformity_score * 0.35
        +
        saturation_score * 0.30
        +
        edge_score * 0.35
    )

    return score


# ============================================================
#                       钢球检测
# ============================================================

def detect_steel_ball(frame):

    frame_h, frame_w = frame.shape[:2]

    # ========================================================
    #                         ROI
    # ========================================================

    x1 = clamp(
        PIPE_X1,
        0,
        frame_w - 1
    )

    y1 = clamp(
        PIPE_Y1,
        0,
        frame_h - 1
    )

    x2 = clamp(
        PIPE_X2,
        0,
        frame_w
    )

    y2 = clamp(
        PIPE_Y2,
        0,
        frame_h
    )

    if x2 <= x1 or y2 <= y1:
        return None

    pipe = frame[
        y1:y2,
        x1:x2
    ]

    if pipe.size == 0:
        return None

    # ========================================================
    #                         高斯模糊
    # ========================================================

    blurred = cv2.GaussianBlur(
        pipe,
        (7, 7),
        1.5
    )

    # ========================================================
    #                         灰度
    # ========================================================

    gray = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2GRAY
    )

    # ========================================================
    #                         HSV
    # ========================================================

    hsv = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV
    )

    # ========================================================
    #                         霍夫圆
    # ========================================================

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=MIN_CIRCLE_DISTANCE,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS
    )

    if circles is None:
        return None

    circles = np.round(
        circles[0]
    ).astype(int)

    candidates = []

    # ========================================================
    #                   遍历所有候选圆
    # ========================================================

    for circle in circles:

        cx, cy, r = circle

        if (
            cx - r < 0
            or
            cy - r < 0
            or
            cx + r >= pipe.shape[1]
            or
            cy + r >= pipe.shape[0]
        ):
            continue

        score = calculate_ball_score(
            gray,
            hsv,
            cx,
            cy,
            r
        )

        if score < 0:
            continue

        # 转换成原始画面的坐标
        global_x = cx + x1
        global_y = cy + y1

        candidates.append(
            (
                global_x,
                global_y,
                r,
                score
            )
        )

    # ========================================================
    #                     没有候选
    # ========================================================

    if not candidates:
        return None

    # ========================================================
    #                     按评分排序
    # ========================================================

    candidates.sort(
        key=lambda item: item[3],
        reverse=True
    )

    # ========================================================
    #        如果上一帧已经找到钢球
    #        优先寻找附近的钢球
    # ========================================================

    if tracker.ball_position is not None:

        previous_x, previous_y = (
            tracker.ball_position
        )

        best_candidate = None

        best_value = -999.0

        for candidate in candidates:

            cx, cy, r, score = candidate

            d = point_distance(
                (cx, cy),
                (previous_x, previous_y)
            )

            if d > MAX_TRACK_DISTANCE:
                continue

            distance_score = clamp(
                1.0 -
                d / MAX_TRACK_DISTANCE,
                0.0,
                1.0
            )

            total_score = (
                score * 0.55
                +
                distance_score * 0.45
            )

            if total_score > best_value:

                best_value = total_score

                best_candidate = candidate

        if best_candidate is not None:

            return best_candidate

    # ========================================================
    #                  第一次寻找钢球
    # ========================================================

    return candidates[0]


# ============================================================
#                       钢球跟踪器
# ============================================================

class BallTracker:

    def __init__(self):

        # ====================================================
        #                   钢球基本信息
        # ====================================================

        self.ball_position = None

        self.radius = None

        self.score = 0.0

        self.missing_frames = 0

        # ====================================================
        #                     历史数据
        # ====================================================

        self.position_history = []

        self.time_history = []

        # ====================================================
        #                     状态
        # ====================================================

        self.state = "WAITING"

        # ====================================================
        #                     物理量
        # ====================================================

        # 钢球相对于管道中心的位置
        #
        # 左边：负数
        # 中心：0
        # 右边：正数
        #
        self.position_cm = 0.0

        # 钢球距离管道中心的距离
        #
        # 永远为正数
        #
        self.distance_to_center_cm = 0.0

        # 钢球速度
        #
        # 单位：cm/s
        #
        self.speed_cm_s = 0.0

        # ====================================================
        #                   中心左右
        # ====================================================

        self.center_side = "CENTER"

        # ====================================================
        #                     速度历史
        # ====================================================

        self.speed_history = []

        # ====================================================
        #                  上一次更新时间
        # ====================================================

        self.last_update_time = time.time()

    # ========================================================
    #                       更新
    # ========================================================

    def update(self, detection):

        current_time = time.time()

        # ====================================================
        #                 没有检测到钢球
        # ====================================================

        if detection is None:

            self.missing_frames += 1

            if (
                self.missing_frames
                <= MAX_MISSING_FRAMES
            ):

                self.state = "SEARCHING"

            else:

                self.ball_position = None

                self.radius = None

                self.score = 0.0

                self.position_history.clear()

                self.time_history.clear()

                self.speed_history.clear()

                self.speed_cm_s = 0.0

                self.position_cm = 0.0

                self.distance_to_center_cm = 0.0

                self.center_side = "CENTER"

                self.state = "LOST"

            self.last_update_time = current_time

            return

        # ====================================================
        #                  检测到钢球
        # ====================================================

        x, y, r, score = detection

        new_position = (
            float(x),
            float(y)
        )

        # ====================================================
        #                  第一次检测
        # ====================================================

        if self.ball_position is None:

            self.ball_position = new_position

            self.radius = r

            self.score = score

            self.position_history.clear()

            self.time_history.clear()

            self.speed_history.clear()

            self.position_history.append(
                new_position
            )

            self.time_history.append(
                current_time
            )

            self.speed_cm_s = 0.0

            # =================================================
            #    计算钢球相对于中心的位置
            # =================================================

            self.position_cm = pixel_to_cm(
                new_position[0]
            )

            # =================================================
            #    计算钢球距离中心的绝对距离
            # =================================================

            self.distance_to_center_cm = (
                calculate_distance_to_center(
                    new_position[0]
                )
            )

            self.center_side = (
                get_center_side(
                    new_position[0]
                )
            )

            self.missing_frames = 0

            self.state = "DETECTED"

            self.last_update_time = current_time

            return

        # ====================================================
        #              当前帧和上一帧距离
        # ====================================================

        d = point_distance(
            new_position,
            self.ball_position
        )

        # ====================================================
        #                    防止跳点
        # ====================================================

        if d > MAX_TRACK_DISTANCE:

            self.missing_frames += 1

            if (
                self.missing_frames
                > MAX_MISSING_FRAMES
            ):

                self.ball_position = None

                self.radius = None

                self.score = 0.0

                self.position_history.clear()

                self.time_history.clear()

                self.speed_history.clear()

                self.speed_cm_s = 0.0

                self.position_cm = 0.0

                self.distance_to_center_cm = 0.0

                self.center_side = "CENTER"

                self.state = "LOST"

            else:

                self.state = "SEARCHING"

            self.last_update_time = current_time

            return

        # ====================================================
        #                     正常更新
        # ====================================================

        old_position = self.ball_position

        old_time = self.last_update_time

        self.ball_position = new_position

        self.radius = r

        self.score = score

        self.missing_frames = 0

        # ====================================================
        #                     保存历史
        # ====================================================

        self.position_history.append(
            new_position
        )

        self.time_history.append(
            current_time
        )

        if (
            len(self.position_history)
            >
            MOTION_HISTORY_LENGTH
        ):

            self.position_history.pop(0)

        if (
            len(self.time_history)
            >
            MOTION_HISTORY_LENGTH
        ):

            self.time_history.pop(0)

        # ====================================================
        #         钢球相对于管道中心的位置
        # ====================================================

        self.position_cm = pixel_to_cm(
            new_position[0]
        )

        # ====================================================
        #              钢球距离管道中心
        # ====================================================

        self.distance_to_center_cm = (
            calculate_distance_to_center(
                new_position[0]
            )
        )

        # ====================================================
        #                   中心哪一侧
        # ====================================================

        self.center_side = (
            get_center_side(
                new_position[0]
            )
        )

        # ====================================================
        #                     计算速度
        # ====================================================
        #
        # 这里计算的是沿着管道方向的速度。
        #
        # dx > 0：
        #       向右
        #
        # dx < 0：
        #       向左
        #
        # ====================================================

        dt = (
            current_time
            -
            old_time
        )

        pipe_pixel_length = (
            PIPE_END_X
            -
            PIPE_START_X
        )

        if (
            dt > 0
            and
            pipe_pixel_length > 0
        ):

            dx_pixel = (
                new_position[0]
                -
                old_position[0]
            )

            # 像素 -> cm
            dx_cm = (
                dx_pixel
                /
                pipe_pixel_length
                *
                PIPE_LENGTH_CM
            )

            # cm/s
            instant_speed = (
                dx_cm
                /
                dt
            )

            # =================================================
            #                 保存速度历史
            # =================================================

            self.speed_history.append(
                instant_speed
            )

            if (
                len(self.speed_history)
                >
                SPEED_HISTORY_LENGTH
            ):

                self.speed_history.pop(0)

            # =================================================
            #                  平均速度
            # =================================================

            if self.speed_history:

                self.speed_cm_s = (
                    sum(
                        self.speed_history
                    )
                    /
                    len(
                        self.speed_history
                    )
                )

        # ====================================================
        #                   更新运动状态
        # ====================================================

        self.update_motion_state()

        self.last_update_time = current_time

    # ========================================================
    #                    判断运动状态
    # ========================================================

    def update_motion_state(self):

        if len(
            self.position_history
        ) < 2:

            self.state = "DETECTED"

            return

        first = (
            self.position_history[0]
        )

        last = (
            self.position_history[-1]
        )

        dx = last[0] - first[0]

        dy = last[1] - first[1]

        moved_distance = math.sqrt(
            dx * dx +
            dy * dy
        )

        # ====================================================
        #                       没移动
        # ====================================================

        if (
            moved_distance
            <
            MOVEMENT_THRESHOLD
        ):

            self.state = "STOPPED"

            return

        # ====================================================
        #                判断主要运动方向
        # ====================================================

        if abs(dx) > abs(dy):

            if dx > 0:

                self.state = "RIGHT"

            else:

                self.state = "LEFT"

        else:

            if dy > 0:

                self.state = "DOWN"

            else:

                self.state = "UP"

    # ========================================================
    #                       获取信息
    # ========================================================

    def get_info(self):

        return {
            "position":
                self.ball_position,

            "radius":
                self.radius,

            "score":
                self.score,

            "state":
                self.state,

            "missing":
                self.missing_frames,

            "speed_cm_s":
                self.speed_cm_s,

            "position_cm":
                self.position_cm,

            "distance_to_center_cm":
                self.distance_to_center_cm,

            "center_side":
                self.center_side
        }


# ============================================================
#                     全局 Tracker
# ============================================================

tracker = BallTracker()


# ============================================================
#                    绘制实时画面
# ============================================================

def draw_result(frame):

    result = frame.copy()

    frame_h, frame_w = result.shape[:2]

    # ========================================================
    #                         ROI
    # ========================================================

    x1 = clamp(
        PIPE_X1,
        0,
        frame_w - 1
    )

    y1 = clamp(
        PIPE_Y1,
        0,
        frame_h - 1
    )

    x2 = clamp(
        PIPE_X2,
        0,
        frame_w - 1
    )

    y2 = clamp(
        PIPE_Y2,
        0,
        frame_h - 1
    )

    # ========================================================
    #                       管道 ROI
    # ========================================================

    cv2.rectangle(
        result,
        (x1, y1),
        (x2, y2),
        (255, 255, 0),
        2
    )

    # ========================================================
    #                       管道中心线
    # ========================================================

    center_x = int(
        (
            PIPE_START_X
            +
            PIPE_END_X
        )
        /
        2
    )

    cv2.line(
        result,
        (center_x, y1),
        (center_x, y2),
        (255, 0, 255),
        2
    )

    # ========================================================
    #                         数据
    # ========================================================

    info = tracker.get_info()

    state = info["state"]

    # ========================================================
    #                         状态
    # ========================================================

    cv2.putText(
        result,
        f"STATE: {state}",
        (20, 40),
        FONT,
        0.8,
        (0, 255, 0),
        2
    )

    # ========================================================
    #                         速度
    # ========================================================

    cv2.putText(
        result,
        f"Speed: "
        f"{info['speed_cm_s']:.2f} cm/s",
        (20, 80),
        FONT,
        0.75,
        (0, 255, 255),
        2
    )

    # ========================================================
    #                相对于管道中心的位置
    # ========================================================

    cv2.putText(
        result,
        f"Position: "
        f"{info['position_cm']:+.2f} cm",
        (20, 115),
        FONT,
        0.75,
        (255, 255, 255),
        2
    )

    # ========================================================
    #                   距离管道中心
    # ========================================================

    cv2.putText(
        result,
        f"Distance to Center: "
        f"{info['distance_to_center_cm']:.2f} cm",
        (20, 150),
        FONT,
        0.7,
        (255, 255, 255),
        2
    )

    # ========================================================
    #                       中心哪边
    # ========================================================

    cv2.putText(
        result,
        f"Side: "
        f"{info['center_side']}",
        (20, 185),
        FONT,
        0.7,
        (255, 255, 255),
        2
    )

    # ========================================================
    #                       钢球
    # ========================================================

    if tracker.ball_position is not None:

        x, y = tracker.ball_position

        x = int(x)

        y = int(y)

        if tracker.radius is not None:

            r = int(
                tracker.radius
            )

        else:

            r = 10

        # ====================================================
        #                    钢球外圈
        # ====================================================

        cv2.circle(
            result,
            (x, y),
            r,
            (0, 255, 0),
            3
        )

        # ====================================================
        #                    钢球中心
        # ====================================================

        cv2.circle(
            result,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )

        # ====================================================
        #                       十字
        # ====================================================

        cv2.line(
            result,
            (x - 20, y),
            (x + 20, y),
            (0, 255, 0),
            2
        )

        cv2.line(
            result,
            (x, y - 20),
            (x, y + 20),
            (0, 255, 0),
            2
        )

        # ====================================================
        #                    像素坐标
        # ====================================================

        cv2.putText(
            result,
            f"Ball: ({x}, {y})",
            (x + 15, y - 15),
            FONT,
            0.55,
            (0, 255, 0),
            2
        )

        # ====================================================
        #                       半径
        # ====================================================

        cv2.putText(
            result,
            f"R: {r}",
            (x + 15, y + 10),
            FONT,
            0.55,
            (0, 255, 0),
            2
        )

        # ====================================================
        #                       评分
        # ====================================================

        cv2.putText(
            result,
            f"Score: "
            f"{tracker.score:.2f}",
            (x + 15, y + 35),
            FONT,
            0.55,
            (0, 255, 0),
            2
        )

        # ====================================================
        #                   中心位置
        # ====================================================

        cv2.putText(
            result,
            f"Pos: "
            f"{tracker.position_cm:+.2f} cm",
            (x + 15, y + 60),
            FONT,
            0.6,
            (0, 255, 255),
            2
        )

        # ====================================================
        #                   距离中心
        # ====================================================

        cv2.putText(
            result,
            f"Center: "
            f"{tracker.distance_to_center_cm:.2f} cm",
            (x + 15, y + 85),
            FONT,
            0.6,
            (0, 255, 255),
            2
        )

    # ========================================================
    #                       搜索中
    # ========================================================

    if state == "SEARCHING":

        cv2.putText(
            result,
            f"Searching... "
            f"{tracker.missing_frames}",
            (20, 220),
            FONT,
            0.7,
            (0, 255, 255),
            2
        )

    # ========================================================
    #                       丢失
    # ========================================================

    elif state == "LOST":

        cv2.putText(
            result,
            "BALL LOST",
            (20, 220),
            FONT,
            0.8,
            (0, 0, 255),
            2
        )

    return result


# ============================================================
#                     DEBUG 候选圆
# ============================================================

def draw_debug_candidates(frame):

    frame_h, frame_w = frame.shape[:2]

    x1 = clamp(
        PIPE_X1,
        0,
        frame_w - 1
    )

    y1 = clamp(
        PIPE_Y1,
        0,
        frame_h - 1
    )

    x2 = clamp(
        PIPE_X2,
        0,
        frame_w
    )

    y2 = clamp(
        PIPE_Y2,
        0,
        frame_h
    )

    pipe = frame[
        y1:y2,
        x1:x2
    ]

    if pipe.size == 0:

        return frame.copy()

    debug = pipe.copy()

    # ========================================================
    #                         灰度
    # ========================================================

    gray = cv2.cvtColor(
        pipe,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (7, 7),
        1.5
    )

    # ========================================================
    #                         HSV
    # ========================================================

    hsv = cv2.cvtColor(
        pipe,
        cv2.COLOR_BGR2HSV
    )

    # ========================================================
    #                         霍夫圆
    # ========================================================

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=MIN_CIRCLE_DISTANCE,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS
    )

    if circles is None:

        cv2.putText(
            debug,
            "NO CIRCLE",
            (20, 40),
            FONT,
            0.8,
            (0, 0, 255),
            2
        )

        return debug

    circles = np.round(
        circles[0]
    ).astype(int)

    # ========================================================
    #                    绘制候选圆
    # ========================================================

    for cx, cy, r in circles:

        if (
            cx - r < 0
            or
            cy - r < 0
            or
            cx + r >= pipe.shape[1]
            or
            cy + r >= pipe.shape[0]
        ):
            continue

        score = calculate_ball_score(
            gray,
            hsv,
            cx,
            cy,
            r
        )

        # 候选圆
        cv2.circle(
            debug,
            (cx, cy),
            r,
            (0, 165, 255),
            2
        )

        # 中心
        cv2.circle(
            debug,
            (cx, cy),
            2,
            (0, 0, 255),
            -1
        )

        # 评分
        cv2.putText(
            debug,
            f"{score:.2f}",
            (cx, cy),
            FONT,
            0.5,
            (255, 255, 255),
            1
        )

    return debug


# ============================================================
#                     控制台实时输出
# ============================================================

def print_status():

    info = tracker.get_info()

    text = (
        f"\r"
        f"状态: {info['state']:<10} | "
        f"中心位置: "
        f"{info['position_cm']:+.2f} cm | "
        f"速度: "
        f"{info['speed_cm_s']:.2f} cm/s | "
        f"距中心: "
        f"{info['distance_to_center_cm']:.2f} cm | "
        f"评分: "
        f"{info['score']:.2f} | "
        f"丢失帧: "
        f"{info['missing']}"
    )

    print(
        text,
        end="",
        flush=True
    )


# ============================================================
#                         主程序
# ============================================================

def main():

    global tracker
    global DEBUG_MODE

    # ========================================================
    #                       基本信息
    # ========================================================

    print("=" * 60)

    print(
        "钢球实时检测 + 位置 + 速度系统"
    )

    print("=" * 60)

    print()

    print(
        f"视频地址: {VIDEO_URL}"
    )

    print(
        f"管道实际长度: "
        f"{PIPE_LENGTH_CM:.2f} cm"
    )

    print(
        f"管道左端 X: "
        f"{PIPE_START_X}"
    )

    print(
        f"管道右端 X: "
        f"{PIPE_END_X}"
    )

    # ========================================================
    #                   管道像素长度
    # ========================================================

    pipe_pixel_length = (
        PIPE_END_X
        -
        PIPE_START_X
    )

    print(
        f"管道像素长度: "
        f"{pipe_pixel_length:.2f} pixel"
    )

    if pipe_pixel_length > 0:

        print(
            f"像素比例: "
            f"{pipe_pixel_length / PIPE_LENGTH_CM:.2f} "
            f"pixel/cm"
        )

    # ========================================================
    #                       管道中心
    # ========================================================

    pipe_center_x = (
        PIPE_START_X
        +
        PIPE_END_X
    ) / 2.0

    print(
        f"管道中心 X: "
        f"{pipe_center_x:.2f}"
    )

    print()

    print(
        "位置坐标："
    )

    print(
        "左端 = -12.50 cm"
    )

    print(
        "中心 = 0.00 cm"
    )

    print(
        "右端 = +12.50 cm"
    )

    print()

    print(
        "按键："
    )

    print(
        "Q / ESC : 退出"
    )

    print(
        "R       : 重置钢球跟踪"
    )

    print(
        "D       : 开关 DEBUG"
    )

    print()

    # ========================================================
    #                    连接手机摄像头
    # ========================================================

    print(
        "正在连接手机摄像头..."
    )

    cap = cv2.VideoCapture(
        VIDEO_URL
    )

    try:

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

    except:

        pass

    if not cap.isOpened():

        print()

        print(
            "无法连接手机摄像头！"
        )

        print()

        print(
            "请检查："
        )

        print(
            "1. 手机 IP 是否正确"
        )

        print(
            "2. 手机和电脑是否在同一个网络"
        )

        print(
            "3. 手机端摄像头服务是否启动"
        )

        print(
            "4. 端口是否为 4747"
        )

        return

    print(
        "手机摄像头连接成功！"
    )

    print()

    # ========================================================
    #                         FPS
    # ========================================================

    fps_counter = 0

    fps_start_time = time.time()

    current_fps = 0.0

    last_print_time = time.time()

    # ========================================================
    #                       主循环
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "\n视频帧读取失败，正在重试..."
            )

            time.sleep(0.1)

            continue

        # ====================================================
        #                         FPS
        # ====================================================

        fps_counter += 1

        current_time = time.time()

        elapsed = (
            current_time
            -
            fps_start_time
        )

        if elapsed >= 1.0:

            current_fps = (
                fps_counter
                /
                elapsed
            )

            fps_counter = 0

            fps_start_time = current_time

        # ====================================================
        #                       检测钢球
        # ====================================================

        detection = detect_steel_ball(
            frame
        )

        # ====================================================
        #                       更新 Tracker
        # ====================================================

        tracker.update(
            detection
        )

        # ====================================================
        #                       绘制结果
        # ====================================================

        display = draw_result(
            frame
        )

        # ====================================================
        #                       FPS 显示
        # ====================================================

        cv2.putText(
            display,
            f"FPS: {current_fps:.1f}",
            (20, 255),
            FONT,
            0.7,
            (255, 255, 255),
            2
        )

        # ====================================================
        #                       主画面
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        # ====================================================
        #                       DEBUG
        # ====================================================

        if DEBUG_MODE:

            debug_frame = (
                draw_debug_candidates(
                    frame
                )
            )

            cv2.imshow(
                "DEBUG - Circle Candidates",
                debug_frame
            )

        # ====================================================
        #                     控制台输出
        # ====================================================

        if (
            time.time()
            -
            last_print_time
            >
            0.1
        ):

            print_status()

            last_print_time = time.time()

        # ====================================================
        #                       键盘
        # ====================================================

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )

        # ====================================================
        #                       Q / ESC
        # ====================================================

        if (
            key == ord("q")
            or
            key == 27
        ):

            break

        # ====================================================
        #                          R
        # ====================================================

        elif key == ord("r"):

            tracker = BallTracker()

            print(
                "\n已重置钢球跟踪器"
            )

        # ====================================================
        #                          D
        # ====================================================

        elif key == ord("d"):

            DEBUG_MODE = (
                not DEBUG_MODE
            )

            if not DEBUG_MODE:

                try:

                    cv2.destroyWindow(
                        "DEBUG - Circle Candidates"
                    )

                except:

                    pass

            print(
                f"\nDEBUG_MODE = "
                f"{DEBUG_MODE}"
            )

    # ========================================================
    #                       释放资源
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    print()

    print()

    print("=" * 60)

    print(
        "程序结束"
    )

    print("=" * 60)


# ============================================================
#                         程序入口
# ============================================================

if __name__ == "__main__":

    main()