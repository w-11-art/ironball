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
# 只用于自动识别绿色管道。
# 钢球判断部分不使用钢球 HSV。
# ============================================================

PIPE_H_LOW = 50
PIPE_H_HIGH = 90
PIPE_S_LOW = 40
PIPE_S_HIGH = 180
PIPE_V_LOW = 25
PIPE_V_HIGH = 200


# ============================================================
#              5. 管道形状判断
#
# 绿色目标应该是一个长条状、接近矩形的管道。
# ============================================================

MIN_PIPE_AREA = 3000
MIN_PIPE_LONG_SIDE = 300
MIN_PIPE_ASPECT = 3.5
MIN_PIPE_RECTANGULARITY = 0.50
MIN_PIPE_SOLIDITY = 0.65
PIPE_BOX_UPDATE_INTERVAL = 8
PIPE_BOX_SMOOTH_ALPHA = 0.20
PIPE_LOST_LIMIT = 12

# 管道内部“无钢球”判定参数
NO_BALL_GREEN_RATIO = 0.82
NO_BALL_GREEN_STD_MAX = 38.0
NO_BALL_NON_GREEN_RATIO_MAX = 0.08


# ============================================================
#              6. 钢球大小约束
#
# 保持原来的钢球思路：
# 非管道 + 面积 + 圆形度 + 半径 + 宽高比 + 跟踪位置
# 不使用钢球 HSV。
# ============================================================

MIN_BALL_AREA = 20
MAX_BALL_AREA = 1800
MIN_BALL_RADIUS = 4
MAX_BALL_RADIUS = 35
MIN_CIRCULARITY = 0.30
MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 2.20

BALL_VERTICAL_RATIO_TOP = 0.15
BALL_VERTICAL_RATIO_BOTTOM = 0.85


# ============================================================
#              7. 锁定参数
# ============================================================

TRACK_SEARCH_WIDTH = 180
TRACK_SEARCH_HEIGHT = 110
LOST_SEARCH_WIDTH = 300
LOST_SEARCH_HEIGHT = 150
MAX_LOST_FRAMES = 8


# ============================================================
#              8. 加速度误判判断
#
# 速度仍然计算，用于预测。
# 误判判断改成：
#     单帧位置跳跃过大 OR 加速度过大
# 不再直接用速度过大判断误判。
# ============================================================

MAX_SINGLE_JUMP_CM = 2.0
MAX_REASONABLE_ACCELERATION_CM_S2 = 1200.0
VELOCITY_ALPHA = 0.20


# ============================================================
#              9. FPS
# ============================================================


# ============================================================
#                  最新帧摄像头
# ============================================================

class LatestFrameCamera:

    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
            return self.frame.copy(), self.frame_id

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


# ============================================================
#                    工具函数
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def resize_frame(frame):
    h, w = frame.shape[:2]
    if w <= PROCESS_WIDTH:
        return frame
    scale = PROCESS_WIDTH / w
    return cv2.resize(
        frame,
        (PROCESS_WIDTH, int(h * scale)),
        interpolation=cv2.INTER_AREA
    )


def order_box_points(points):
    """返回 TL, TR, BR, BL 顺序。"""
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def box_center(box_points):
    return np.mean(box_points, axis=0)


def box_long_direction(box_points):
    """返回矩形长边方向（单位向量）。"""
    pts = order_box_points(box_points)
    top = pts[1] - pts[0]
    side = pts[2] - pts[1]

    if np.linalg.norm(top) >= np.linalg.norm(side):
        direction = top
    else:
        direction = pts[2] - pts[1]

    norm = np.linalg.norm(direction)
    if norm <= 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)

    direction = direction / norm
    if direction[0] < 0:
        direction = -direction
    return direction


def blend_box(old_box, new_box, alpha):
    old_pts = np.asarray(old_box, dtype=np.float32)
    new_pts = np.asarray(new_box, dtype=np.float32)

    old_pts = order_box_points(old_pts)
    new_pts = order_box_points(new_pts)

    blended = (
        old_pts * (1.0 - alpha)
        + new_pts * alpha
    )
    return blended.astype(np.float32)


def pipe_model_from_box(box_points):
    """生成供钢球模块使用的 pipe model。"""
    pts = order_box_points(box_points)

    width = np.linalg.norm(pts[1] - pts[0])
    height = np.linalg.norm(pts[3] - pts[0])

    if height > width:
        # 保证 width 是长边
        pts = np.array([pts[3], pts[0], pts[1], pts[2]], dtype=np.float32)
        width = np.linalg.norm(pts[1] - pts[0])
        height = np.linalg.norm(pts[3] - pts[0])

    center = box_center(pts)
    direction = box_long_direction(pts)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)

    # 确保法向量朝向矩形短边方向
    normal_norm = np.linalg.norm(normal)
    if normal_norm > 1e-6:
        normal = normal / normal_norm

    return {
        "box": pts,
        "center": center,
        "direction": direction,
        "normal": normal,
        "length_px": float(width),
        "width_px": float(height),
    }


def is_quad_like_pipe(contour, box_points):
    """判断绿色目标是否足够像二维长方形/长圆柱投影。"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 0 or perimeter <= 0:
        return False, {"reason": "empty_contour"}

    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    if rw <= 1 or rh <= 1:
        return False, {"reason": "invalid_rect"}

    long_side = max(rw, rh)
    short_side = min(rw, rh)
    aspect = long_side / max(short_side, 1e-6)

    rect_area = rw * rh
    rectangularity = area / max(rect_area, 1.0)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1.0)

    approx = cv2.approxPolyDP(
        contour,
        0.025 * perimeter,
        True
    )
    vertices = len(approx)

    # 长条矩形的合理顶点数通常不应该是非常复杂的多边形。
    # 允许 3 个顶点: 管道一端被球/画面边缘遮挡时, 多边形会合并成三角。
    vertex_ok = 3 <= vertices <= 12

    valid = (
        area >= MIN_PIPE_AREA
        and long_side >= MIN_PIPE_LONG_SIDE
        and aspect >= MIN_PIPE_ASPECT
        and rectangularity >= MIN_PIPE_RECTANGULARITY
        and solidity >= MIN_PIPE_SOLIDITY
        and vertex_ok
    )

    info = {
        "area": area,
        "long_side": long_side,
        "short_side": short_side,
        "aspect": aspect,
        "rectangularity": rectangularity,
        "solidity": solidity,
        "vertices": vertices,
        "valid": valid,
    }
    return valid, info


# ============================================================
#                自动识别绿色管道
# ============================================================

def find_pipe_model(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array(
        [PIPE_H_LOW, PIPE_S_LOW, PIPE_V_LOW],
        dtype=np.uint8
    )
    upper = np.array(
        [PIPE_H_HIGH, PIPE_S_HIGH, PIPE_V_HIGH],
        dtype=np.uint8
    )

    mask = cv2.inRange(hsv, lower, upper)

    # 保持你之前绿色管道的长条特征
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (31, 7)
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_open
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_PIPE_AREA:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = order_box_points(box)

        valid, info = is_quad_like_pipe(contour, box)
        info["contour"] = contour
        info["box"] = box

        if valid:
            # 面积 + 长宽比综合排序
            score = (
                min(info["aspect"] / 8.0, 1.0) * 0.35
                + min(info["rectangularity"], 1.0) * 0.35
                + min(info["solidity"], 1.0) * 0.15
                + min(info["area"] / 20000.0, 1.0) * 0.15
            )
            candidates.append((score, info))

    if not candidates:
        # 返回最近失败信息，供控制台 debug
        largest_debug = None
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            long_side = max(rw, rh)
            short_side = min(rw, rh)
            aspect = long_side / max(short_side, 1e-6)
            rect_area = rw * rh
            rectangularity = area / max(rect_area, 1.0)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1.0)
            approx = cv2.approxPolyDP(
                contour,
                0.025 * cv2.arcLength(contour, True),
                True
            )
            largest_debug = {
                "area": area,
                "aspect": aspect,
                "rectangularity": rectangularity,
                "solidity": solidity,
                "vertices": len(approx),
            }

        return None, largest_debug

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return candidates[0][1], None


# ============================================================
#             钢球原有检测逻辑
#
# 关键：这里只接受 pipe_model 作为“管道建模区域”。
# 钢球本身不使用 HSV。
# ============================================================

def get_ball_zone(pipe_model):
    box = order_box_points(pipe_model["box"])
    center = pipe_model["center"]
    direction = pipe_model["direction"]
    normal = pipe_model["normal"]
    length = pipe_model["length_px"]
    width = pipe_model["width_px"]

    # 中间 70% 作为钢球搜索区域
    half_len = length * 0.50
    half_w = width * 0.5
    top_margin = half_w * (1.0 - BALL_VERTICAL_RATIO_BOTTOM)
    bottom_margin = half_w * (1.0 - BALL_VERTICAL_RATIO_TOP)

    # 用一个旋转矩形的四角描述 zone
    left = center - direction * half_len
    right = center + direction * half_len
    top = left - normal * top_margin
    bottom = left + normal * bottom_margin
    top2 = right - normal * top_margin
    bottom2 = right + normal * bottom_margin

    poly = np.array(
        [top, top2, bottom2, bottom],
        dtype=np.float32
    )

    return poly


def pipe_crop(frame, polygon):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(polygon).astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask


def analyze_pipe_fill(frame, pipe_model):
    """判断自动识别出的管道框内部是否基本均匀为绿色。"""
    pts = order_box_points(pipe_model["box"]).astype(np.int32)

    min_x = max(0, int(np.min(pts[:, 0])) - 1)
    max_x = min(frame.shape[1], int(np.max(pts[:, 0])) + 2)
    min_y = max(0, int(np.min(pts[:, 1])) - 1)
    max_y = min(frame.shape[0], int(np.max(pts[:, 1])) + 2)

    if max_x <= min_x or max_y <= min_y:
        return True, 0.0, 0.0, 1.0

    roi = frame[min_y:max_y, min_x:max_x]
    local_poly = pts - np.array([min_x, min_y], dtype=np.int32)

    polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(polygon_mask, local_poly, 255)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([PIPE_H_LOW, PIPE_S_LOW, PIPE_V_LOW], dtype=np.uint8)
    upper = np.array([PIPE_H_HIGH, PIPE_S_HIGH, PIPE_V_HIGH], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower, upper)

    inside = polygon_mask > 0
    inside_count = int(np.count_nonzero(inside))
    if inside_count == 0:
        return True, 0.0, 0.0, 1.0

    green_pixels = green_mask[inside] > 0
    green_ratio = float(green_pixels.mean())
    non_green_ratio = 1.0 - green_ratio

    v_values = hsv[:, :, 2][inside]
    green_v_values = v_values[green_pixels]
    if green_v_values.size > 10:
        green_v_std = float(np.std(green_v_values))
    else:
        green_v_std = 999.0

    no_ball_uniform_green = (
        green_ratio >= NO_BALL_GREEN_RATIO
        and green_v_std <= NO_BALL_GREEN_STD_MAX
        and non_green_ratio <= NO_BALL_NON_GREEN_RATIO_MAX
    )

    return no_ball_uniform_green, green_ratio, green_v_std, non_green_ratio


def create_non_green_inside_pipe(frame, pipe_model):
    pipe_polygon = np.asarray(pipe_model["box"], dtype=np.float32)
    pad = 3

    min_x = max(0, int(np.floor(np.min(pipe_polygon[:, 0]))) - pad)
    max_x = min(frame.shape[1], int(np.ceil(np.max(pipe_polygon[:, 0]))) + pad + 1)
    min_y = max(0, int(np.floor(np.min(pipe_polygon[:, 1]))) - pad)
    max_y = min(frame.shape[0], int(np.ceil(np.max(pipe_polygon[:, 1]))) + pad + 1)

    pw = max_x - min_x
    ph = max_y - min_y
    if pw <= 0 or ph <= 0:
        return None

    roi = frame[min_y:max_y, min_x:max_x]
    local_polygon = (
        np.round(pipe_polygon - np.array([min_x, min_y], dtype=np.float32))
        .astype(np.int32)
    )

    zone_mask = np.zeros((ph, pw), dtype=np.uint8)
    cv2.fillConvexPoly(zone_mask, local_polygon, 255)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([PIPE_H_LOW, PIPE_S_LOW, PIPE_V_LOW], dtype=np.uint8)
    upper = np.array([PIPE_H_HIGH, PIPE_S_HIGH, PIPE_V_HIGH], dtype=np.uint8)
    pipe_green = cv2.inRange(hsv, lower, upper)

    non_green = cv2.bitwise_not(pipe_green)
    non_green = cv2.bitwise_and(non_green, zone_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    non_green = cv2.morphologyEx(non_green, cv2.MORPH_OPEN, kernel)

    return non_green, min_x, min_y, hsv, zone_mask


def circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return 4.0 * np.pi * area / (perimeter * perimeter)


def green_surround_score(hsv, x, y, radius):
    h, w = hsv.shape[:2]
    outer = max(int(radius * 2.5), 10)
    inner = max(int(radius * 1.3), 5)

    x1 = max(0, int(x - outer))
    y1 = max(0, int(y - outer))
    x2 = min(w, int(x + outer + 1))
    y2 = min(h, int(y + outer + 1))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    local = hsv[y1:y2, x1:x2]
    yy, xx = np.ogrid[y1:y2, x1:x2]
    distance = np.sqrt(
        (xx - x) ** 2 +
        (yy - y) ** 2
    )

    ring = (
        (distance >= inner) &
        (distance <= outer)
    )

    if not np.any(ring):
        return 0.0

    H = local[:, :, 0]
    S = local[:, :, 1]
    V = local[:, :, 2]

    green = (
        (H >= PIPE_H_LOW) &
        (H <= PIPE_H_HIGH) &
        (S >= PIPE_S_LOW) &
        (V >= PIPE_V_LOW)
    )

    return float(green[ring].mean())


def find_ball_candidates(frame, pipe_model):
    no_ball_uniform_green, green_ratio, green_v_std, non_green_ratio = analyze_pipe_fill(
        frame, pipe_model
    )

    scene_info = {
        "no_ball_uniform_green": no_ball_uniform_green,
        "green_ratio": green_ratio,
        "green_v_std": green_v_std,
        "non_green_ratio": non_green_ratio,
    }

    # 关键：如果整个管道框内部基本都是稳定绿色，
    # 直接判定“没有钢球”，绝不从噪声里随便挑一个圆。
    if no_ball_uniform_green:
        return [], scene_info

    result = create_non_green_inside_pipe(frame, pipe_model)
    if result is None:
        return [], scene_info

    non_green, offset_x, offset_y, hsv, zone_mask = result

    contours, _ = cv2.findContours(
        non_green,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_BALL_AREA or area > MAX_BALL_AREA:
            continue

        shape = circularity(contour)
        if shape < MIN_CIRCULARITY:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw <= 0 or bh <= 0:
            continue

        aspect = bw / bh
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < MIN_BALL_RADIUS or radius > MAX_BALL_RADIUS:
            continue

        global_x = cx + offset_x
        global_y = cy + offset_y

        ring_score = green_surround_score(hsv, cx, cy, radius)
        if ring_score < 0.20:
            continue

        if shape >= 0.75:
            shape_bonus = 1.0
        elif shape >= 0.55:
            shape_bonus = 0.8
        else:
            shape_bonus = 0.5

        score = shape_bonus * 0.50 + min(ring_score, 1.0) * 0.50

        candidates.append({
            "x": global_x,
            "y": global_y,
            "radius": radius,
            "area": area,
            "bbox": (
                bx + offset_x,
                by + offset_y,
                bw,
                bh
            ),
            "circularity": shape,
            "aspect": aspect,
            "ring_score": ring_score,
            "score": score,
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates, scene_info


def choose_ball(candidates, previous_ball, predicted_position, strict=False):
    if not candidates:
        return None

    if previous_ball is None:
        return candidates[0]

    best = None
    best_score = -999.0

    for candidate in candidates:
        dx = candidate["x"] - predicted_position[0]
        dy = candidate["y"] - predicted_position[1]
        distance = math.sqrt(dx * dx + dy * dy)

        max_distance = 60 if strict else 100
        if distance > max_distance:
            continue

        distance_score = max(
            0.0,
            1.0 - distance / max_distance
        )

        old_radius = previous_ball["radius"]
        radius_ratio = candidate["radius"] / max(old_radius, 1)

        if strict:
            if not (0.70 <= radius_ratio <= 1.30):
                continue
        else:
            if not (0.55 <= radius_ratio <= 1.50):
                continue

        radius_score = max(
            0.0,
            1.0 - abs(1.0 - radius_ratio)
        )

        total_score = (
            candidate["score"] * 0.50
            + distance_score * 0.35
            + radius_score * 0.15
        )

        if total_score > best_score:
            best_score = total_score
            best = candidate

    return best


def pixel_to_position_cm(x, pipe_model):
    center = pipe_model["center"]
    direction = pipe_model["direction"]
    length_px = pipe_model["length_px"]

    if length_px <= 1e-6:
        return 0.0

    point = np.array([x, center[1]], dtype=np.float32)
    # 位置只取沿管道轴方向的坐标。
    # y 对这里的中心值只用于构造一个与球 x 相同的点，
    # 后面在主程序里使用二维点版本进行精确计算。
    return 0.0


def point_to_position_cm(point, pipe_model):
    center = pipe_model["center"]
    direction = pipe_model["direction"]
    length_px = pipe_model["length_px"]

    if length_px <= 1e-6:
        return 0.0

    p = np.array(point, dtype=np.float32)
    longitudinal = float(
        np.dot(
            p - center,
            direction
        )
    )

    return (
        longitudinal
        /
        length_px
        *
        PIPE_LENGTH_CM
    )


def calculate_speed(current_point, previous_point, dt, pipe_model):
    if dt <= 0:
        return 0.0

    center = pipe_model["center"]
    direction = pipe_model["direction"]
    length_px = pipe_model["length_px"]

    if length_px <= 1e-6:
        return 0.0

    p1 = np.array(previous_point, dtype=np.float32)
    p2 = np.array(current_point, dtype=np.float32)

    dx_pixel = float(
        np.dot(
            p2 - p1,
            direction
        )
    )

    dx_cm = (
        dx_pixel
        /
        length_px
        *
        PIPE_LENGTH_CM
    )

    return dx_cm / dt


def point_on_centerline(pipe_model, side="center"):
    pts = order_box_points(pipe_model["box"])
    center = pipe_model["center"]
    direction = pipe_model["direction"]
    half_len = pipe_model["length_px"] / 2.0

    if side == "left":
        return center - direction * half_len
    if side == "right":
        return center + direction * half_len
    return center


# ============================================================
#                     主程序
# ============================================================

def main():
    print("=" * 70)
    print("自动识别绿色管道 + 管道建模 + 钢球锁定系统")
    print("=" * 70)
    print(f"视频: {VIDEO_URL}")
    print(f"管道实际长度: {PIPE_LENGTH_CM:.1f} cm")
    print("Q = 退出")
    print("R = 重新识别管道并重新锁球")
    print()

    camera = LatestFrameCamera(VIDEO_URL)
    time.sleep(1.5)

    if not camera.cap.isOpened():
        print("手机摄像头连接失败！")
        camera.stop()
        return

    print("手机摄像头连接成功！")

    # ---------------- 管道状态 ----------------
    pipe_model = None
    pipe_fail_count = 0
    pipe_debug_message = ""
    pipe_debug_until = 0.0
    frame_counter = 0

    # ---------------- 钢球状态 ----------------
    ball = None
    previous_point = None
    previous_time = None
    velocity_cm_s = 0.0
    acceleration_cm_s2 = 0.0
    lost_frames = 0

    # ---------------- 自检状态 ----------------
    verify_mode = False
    verify_count = 0

    # ---------------- FPS ----------------
    process_fps = 0.0
    process_count = 0
    process_start = time.perf_counter()

    camera_fps = 0.0
    camera_count = 0
    last_frame_id = -1
    camera_start = time.perf_counter()

    try:
        while True:
            frame, frame_id = camera.read()
            if frame is None:
                continue

            frame = resize_frame(frame)
            now = time.perf_counter()
            frame_counter += 1

            # =================================================
            # PROCESS FPS
            # =================================================
            process_count += 1
            if now - process_start >= 1.0:
                process_fps = process_count / (now - process_start)
                process_count = 0
                process_start = now

            # =================================================
            # CAMERA FPS
            # =================================================
            if last_frame_id < 0:
                last_frame_id = frame_id
            elif frame_id != last_frame_id:
                camera_count += frame_id - last_frame_id
                last_frame_id = frame_id

            if now - camera_start >= 1.0:
                camera_fps = camera_count / (now - camera_start)
                camera_count = 0
                camera_start = now

            # =================================================
            # 自动识别/更新管道
            # =================================================
            need_pipe_detection = (
                pipe_model is None
                or ball is None
                or frame_counter % PIPE_BOX_UPDATE_INTERVAL == 0
            )

            if need_pipe_detection:
                detected_pipe, debug_info = find_pipe_model(frame)

                if detected_pipe is not None:
                    new_model = pipe_model_from_box(
                        detected_pipe["box"]
                    )

                    if pipe_model is None:
                        pipe_model = new_model
                    else:
                        blended_box = blend_box(
                            pipe_model["box"],
                            new_model["box"],
                            PIPE_BOX_SMOOTH_ALPHA
                        )
                        pipe_model = pipe_model_from_box(
                            blended_box
                        )

                    pipe_fail_count = 0
                    pipe_debug_message = ""
                else:
                    pipe_fail_count += 1

                    if debug_info is not None:
                        pipe_debug_message = (
                            "PIPE DEBUG: "
                            f"A={debug_info.get('area', 0):.0f} "
                            f"AR={debug_info.get('aspect', 0):.2f} "
                            f"Rect={debug_info.get('rectangularity', 0):.2f} "
                            f"Sol={debug_info.get('solidity', 0):.2f} "
                            f"V={debug_info.get('vertices', 0)}"
                        )
                        pipe_debug_until = now + 0.8

                    if pipe_fail_count > PIPE_LOST_LIMIT:
                        pipe_model = None

            # =================================================
            # 如果没有有效管道模型
            # =================================================
            if pipe_model is None:
                cv2.putText(
                    frame,
                    "SEARCHING PIPE...",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                if now < pipe_debug_until and pipe_debug_message:
                    cv2.putText(
                        frame,
                        pipe_debug_message,
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (0, 0, 255),
                        1
                    )

                cv2.putText(
                    frame,
                    f"PROC: {process_fps:.1f} FPS",
                    (frame.shape[1] - 205, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"CAM: {camera_fps:.1f} FPS",
                    (frame.shape[1] - 205, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("Steel Ball Tracking", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("r"):
                    pipe_model = None
                    ball = None
                    previous_point = None
                    previous_time = None
                    velocity_cm_s = 0.0
                    acceleration_cm_s2 = 0.0
                    lost_frames = 0
                    verify_mode = False
                    verify_count = 0

                continue

            # =================================================
            # 钢球预测
            # =================================================
            predicted = None
            if ball is not None:
                dt_predict = (
                    now - previous_time
                    if previous_time is not None
                    else 0.0
                )

                direction = pipe_model["direction"]
                length_px = pipe_model["length_px"]

                pixel_speed = (
                    velocity_cm_s
                    * length_px
                    / PIPE_LENGTH_CM
                )

                predicted_point = (
                    np.array(ball["x_y"], dtype=np.float32)
                    + direction * pixel_speed * dt_predict
                )

                predicted = (
                    float(predicted_point[0]),
                    float(predicted_point[1])
                )

            # =================================================
            # 钢球搜索框
            # =================================================
            if ball is None:
                zone_poly = get_ball_zone(pipe_model)
                search_box = cv2.boundingRect(
                    np.round(zone_poly).astype(np.int32)
                )
                sx, sy, sw, sh = search_box
                search_box = (
                    sx,
                    sy,
                    sx + sw,
                    sy + sh
                )
            else:
                if lost_frames == 0:
                    search_width = TRACK_SEARCH_WIDTH
                    search_height = TRACK_SEARCH_HEIGHT
                else:
                    search_width = LOST_SEARCH_WIDTH
                    search_height = LOST_SEARCH_HEIGHT

                cx, cy = predicted
                search_box = (
                    int(cx - search_width / 2),
                    int(cy - search_height / 2),
                    int(cx + search_width / 2),
                    int(cy + search_height / 2),
                )

            # =================================================
            # 原有钢球候选检测
            # =================================================
            candidates, ball_scene_info = find_ball_candidates(
                frame,
                pipe_model
            )

            no_ball_now = ball_scene_info["no_ball_uniform_green"]

            # 如果管道框内部已经确认是均匀绿色且无明显非绿色物体，
            # 立即显示未找到钢球，清除旧锁定，禁止误圈。
            if no_ball_now:
                if ball is not None:
                    print("[BALL] 管道内部均匀绿色，确认当前没有钢球，解除锁定。")

                ball = None
                previous_point = None
                previous_time = None
                velocity_cm_s = 0.0
                acceleration_cm_s2 = 0.0
                lost_frames = 0
                verify_mode = False
                verify_count = 0

            # 只接受搜索框内候选
            sx1, sy1, sx2, sy2 = search_box
            candidates = [
                c for c in candidates
                if sx1 <= c["x"] <= sx2
                and sy1 <= c["y"] <= sy2
            ]

            if no_ball_now:
                candidate = None
            else:
                candidate = choose_ball(
                    candidates,
                    previous_ball=ball,
                    predicted_position=(
                        predicted
                        if predicted is not None
                        else (0.0, 0.0)
                    ),
                    strict=verify_mode
                )

            # =================================================
            # 自检模式
            # =================================================
            if verify_mode:
                if candidate is None:
                    verify_count += 1
                    if verify_count >= 2:
                        verify_mode = False
                        verify_count = 0
                else:
                    dx = candidate["x"] - ball["x_y"][0]
                    dy = candidate["y"] - ball["x_y"][1]
                    distance = math.hypot(dx, dy)

                    radius_ratio = (
                        candidate["radius"]
                        / max(ball["radius"], 1.0)
                    )

                    radius_ok = 0.70 <= radius_ratio <= 1.30

                    if distance <= 60 and radius_ok:
                        verify_count += 1
                        if verify_count >= 2:
                            ball = {
                                **candidate,
                                "x_y": (
                                    candidate["x"],
                                    candidate["y"]
                                )
                            }
                            previous_point = ball["x_y"]
                            previous_time = now
                            lost_frames = 0
                            verify_mode = False
                            verify_count = 0
                            print("[VERIFY] 重新锁定成功")
                    else:
                        verify_count = 0

            # =================================================
            # 正常跟踪
            # =================================================
            else:
                if candidate is not None:

                    if ball is None:
                        ball = {
                            **candidate,
                            "x_y": (
                                candidate["x"],
                                candidate["y"]
                            )
                        }
                        previous_point = ball["x_y"]
                        previous_time = now
                        velocity_cm_s = 0.0
                        acceleration_cm_s2 = 0.0
                        lost_frames = 0
                        print("[LOCK] 钢球锁定成功")

                    else:
                        current_point = (
                            candidate["x"],
                            candidate["y"]
                        )

                        dt = (
                            now - previous_time
                            if previous_time is not None
                            else 0.0
                        )

                        raw_speed = calculate_speed(
                            current_point,
                            previous_point,
                            dt,
                            pipe_model
                        )

                        new_velocity = (
                            (1.0 - VELOCITY_ALPHA) * velocity_cm_s
                            + VELOCITY_ALPHA * raw_speed
                        )

                        if dt > 0:
                            current_acceleration = (
                                new_velocity - velocity_cm_s
                            ) / dt
                        else:
                            current_acceleration = 0.0

                        jump_cm = abs(
                            point_to_position_cm(
                                current_point,
                                pipe_model
                            )
                            -
                            point_to_position_cm(
                                previous_point,
                                pipe_model
                            )
                        )

                        suspicious = (
                            jump_cm > MAX_SINGLE_JUMP_CM
                            or abs(current_acceleration)
                            > MAX_REASONABLE_ACCELERATION_CM_S2
                        )

                        if suspicious:
                            print()
                            print("[SELF-CHECK] 疑似误判")
                            print(
                                f"单帧位置跳动: {jump_cm:.2f} cm"
                            )
                            print(
                                f"内部速度: {new_velocity:.2f} cm/s"
                            )
                            print(
                                f"内部加速度: "
                                f"{current_acceleration:.2f} cm/s²"
                            )
                            print("不接受这个候选，进入 VERIFY")
                            verify_mode = True
                            verify_count = 0
                        else:
                            velocity_cm_s = new_velocity
                            acceleration_cm_s2 = current_acceleration
                            ball = {
                                **candidate,
                                "x_y": current_point
                            }
                            previous_point = current_point
                            previous_time = now
                            lost_frames = 0

                else:
                    lost_frames += 1

                    if lost_frames > MAX_LOST_FRAMES:
                        print("[LOCK] 钢球长时间丢失，重新搜索")
                        ball = None
                        previous_point = None
                        previous_time = None
                        velocity_cm_s = 0.0
                        acceleration_cm_s2 = 0.0
                        lost_frames = 0

            # =================================================
            #             绘制自动识别的管道框
            # =================================================
            pipe_pts = order_box_points(pipe_model["box"]).astype(np.int32)
            cv2.polylines(
                frame,
                [pipe_pts],
                True,
                (0, 255, 0),
                2
            )

            # 中间辅助线：由自动识别管道矩形直接生成
            left_mid = (
                pipe_pts[0].astype(np.float32)
                + pipe_pts[3].astype(np.float32)
            ) / 2.0
            right_mid = (
                pipe_pts[1].astype(np.float32)
                + pipe_pts[2].astype(np.float32)
            ) / 2.0

            cv2.line(
                frame,
                tuple(np.round(left_mid).astype(int)),
                tuple(np.round(right_mid).astype(int)),
                (255, 0, 0),
                2
            )

            # =================================================
            #                     钢球
            # =================================================
            if ball is not None:
                bx, by = ball["x_y"]
                radius = max(int(ball["radius"]), 6)

                cv2.circle(
                    frame,
                    (int(bx), int(by)),
                    radius,
                    (0, 0, 255),
                    2
                )

                cv2.circle(
                    frame,
                    (int(bx), int(by)),
                    4,
                    (0, 0, 255),
                    -1
                )

                position_cm = point_to_position_cm(
                    (bx, by),
                    pipe_model
                )
                distance_cm = abs(position_cm)

                if verify_mode:
                    state = "VERIFY"
                    color = (0, 255, 255)
                elif lost_frames > 0:
                    state = "TRACKING"
                    color = (0, 255, 255)
                else:
                    state = "LOCKED"
                    color = (0, 255, 0)

                cv2.putText(
                    frame,
                    f"BALL: {state}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Position: {position_cm:+.2f} cm",
                    (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Distance: {distance_cm:.2f} cm",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

            else:
                if no_ball_now:
                    cv2.putText(
                        frame,
                        "BALL: NOT FOUND",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.70,
                        (0, 0, 255),
                        2
                    )
                    cv2.putText(
                        frame,
                        "PIPE: UNIFORM GREEN / NO BALL",
                        (20, 68),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (0, 0, 255),
                        1
                    )
                else:
                    cv2.putText(
                        frame,
                        "BALL: SEARCHING",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.70,
                        (0, 165, 255),
                        2
                    )

            # =================================================
            #              管道模型状态
            # =================================================
            pipe_center = pipe_model["center"]
            pipe_len = pipe_model["length_px"]
            pipe_w = pipe_model["width_px"]

            cv2.putText(
                frame,
                f"PIPE: {pipe_len:.0f}x{pipe_w:.0f}px",
                (20, frame.shape[0] - 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            # =================================================
            #                    FPS
            # =================================================
            cv2.putText(
                frame,
                f"PROC: {process_fps:.1f} FPS",
                (frame.shape[1] - 205, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"CAM: {camera_fps:.1f} FPS",
                (frame.shape[1] - 205, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 0),
                2
            )

            # =================================================
            #                  自动管道 DEBUG
            # =================================================
            if pipe_debug_message and now < pipe_debug_until:
                cv2.putText(
                    frame,
                    pipe_debug_message,
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1
                )

            # =================================================
            #                       显示
            # =================================================
            cv2.imshow(
                "Steel Ball Tracking",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r"):
                print("[USER] 重新识别管道并重新搜索钢球")
                pipe_model = None
                pipe_fail_count = 0
                pipe_debug_message = ""
                ball = None
                previous_point = None
                previous_time = None
                velocity_cm_s = 0.0
                acceleration_cm_s2 = 0.0
                lost_frames = 0
                verify_mode = False
                verify_count = 0

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    print("程序结束")


if __name__ == "__main__":
    main()
