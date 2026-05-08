# 导入MaixCAM核心库：摄像头、显示、图像、神经网络、系统、串口
from maix import camera, display, image, nn, app, uart
import cv2
import numpy as np
from struct import pack

# ====================== 卡尔曼滤波器（仅用于YOLO目标）======================
class KalmanFilter2D:
    def __init__(self, init_x, init_y):
        # 状态向量 [x坐标, x方向速度, y坐标, y方向速度]
        self.state = np.array([[init_x], [0.0], [init_y], [0.0]], dtype=np.float32)
        
        # 状态转移矩阵（匀速运动模型）
        self.F = np.array([[1, 1, 0, 0],[0, 1, 0, 0],[0, 0, 1, 1],[0, 0, 0, 1]], dtype=np.float32)
        
        # 观测矩阵：只观测x、y坐标
        self.H = np.array([[1, 0, 0, 0],[0, 0, 1, 0]], dtype=np.float32)
        
        # 过程噪声协方差
        self.Q = np.eye(4, dtype=np.float32) * 0.072
        # 观测噪声协方差
        self.R = np.eye(2, dtype=np.float32) * 23.0
        # 误差协方差矩阵
        self.P = np.eye(4, dtype=np.float32) * 10.0

    # 卡尔曼预测步骤
    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    # 卡尔曼更新步骤：传入检测到的真实坐标
    def update(self, meas_x, meas_y):
        z = np.array([[meas_x], [meas_y]], dtype=np.float32)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    # 获取滤波后的坐标
    def get_pos(self):
        return int(self.state[0, 0]), int(self.state[2, 0])

# ====================== 串口初始化 ======================
uart_dev = uart.UART("/dev/ttyS0", 115200)

# 串口数据发送：发送坐标偏移量
def send_data(err_x, err_y):
    err_x_h = (err_x >> 8) & 0xff
    err_x_l = err_x & 0xff
    err_y_h = (err_y >> 8) & 0xff
    err_y_l = err_y & 0xff
    data = pack("<BBBBBB", 0xAA, err_x_h, err_x_l, err_y_h, err_y_l, 0x5C)
    uart_dev.write(data)

# ====================== 系统参数 ======================
LOST_FRAME_MAX = 50        # 最大连续丢失帧数
lost_frame_cnt = 0         # 当前连续丢失计数
kf_init_x = 253           # 卡尔曼初始X
kf_init_y = 211           # 卡尔曼初始Y
kf = KalmanFilter2D(kf_init_x, kf_init_y)

crop_padding = 23          # 目标裁剪外扩像素
rect_min_limit = 12        # 最小有效矩形尺寸
std_from_white_rect = True # 漫水填充优化
hires_mode = True          # 高分辨率模式
high_res = 448             # 高分辨率大小
model_path = "/root/models/model_3356.mud"  # YOLO模型路径
cam_buff_num = 2           # 摄像头缓存数

# ====================== 硬件初始化 ======================
detector = nn.YOLOv5(model=model_path, dual_buff=True)

# 根据模式初始化摄像头
if hires_mode:
    cam = camera.Camera(high_res, high_res, detector.input_format(), buff_num=cam_buff_num)
else:
    cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format(), buff_num=cam_buff_num)

disp = display.Display()  # 屏幕初始化

# ====================== 全局坐标变量 ======================
center_pos = [253, 211]     # 屏幕中心（激光中心点）
last_center = center_pos    # 上一帧目标中心
last_center_small = [detector.input_width(), detector.input_height()]

# ====================== 主循环 ======================
while not app.need_exit():
    rect_found = False
    img = cam.read()  # 读取一帧图像

    # 缩放到AI模型输入尺寸
    if hires_mode:
        img_ai = img.resize(detector.input_width(), detector.input_height())
    else:
        img_ai = img

    # ====================== YOLO目标检测 ======================
    objs = detector.detect(img_ai, conf_th=0.25, iou_th=0.45)
    max_idx = -1
    max_s = 0
    # 找到面积最大的目标
    for i, obj in enumerate(objs):
        s = obj.w * obj.h
        if s > max_s:
            max_s = s
            max_idx = i

    kf.predict()  # 卡尔曼每帧必预测
    center_to_send = [0, 0]

    # ====================== 检测到YOLO目标 ======================
    if max_idx >= 0:
        obj = objs[max_idx]
        # 扩展检测框，保证完整裁剪目标
        w = obj.w + crop_padding * 2
        h = obj.h + crop_padding * 2
        w = w + 1 if w % 2 != 0 else w
        h = h + 1 if h % 2 != 0 else h
        x = obj.x - crop_padding
        y = obj.y - crop_padding

        # 防止裁剪框越界
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        if x + w > img_ai.width():
            w = img_ai.width() - x
        if y + h > img_ai.height():
            h = img_ai.height() - y

        # 裁剪目标区域
        crop_ai = img_ai.crop(x, y, w, h)
        crop_ai_rect = [x, y, w, h]
        img_ai_scale = [img.width() / img_ai.width(), img.height() / img_ai.height()]

        # 灰度化处理
        gray = crop_ai.to_format(image.Format.FMT_GRAYSCALE)
        gray_cv = image.image2cv(gray, False, False)

        # 自适应阈值二值化
        binary = cv2.adaptiveThreshold(gray_cv, 255,
                      cv2.ADAPTIVE_THRESH_MEAN_C,
                      cv2.THRESH_BINARY_INV, 23, 16)

        # 漫水填充去噪
        if std_from_white_rect:
            h_bin, w_bin = binary.shape[:2]
            mask = np.zeros((h_bin + 2, w_bin + 2), np.uint8)
            cv2.floodFill(binary, mask, (2, 2), 255, loDiff=15, upDiff=15, flags=4)
            cv2.floodFill(binary, mask, (w_bin - 2, h_bin - 2), 255, loDiff=15, upDiff=15, flags=4)
            binary = cv2.bitwise_not(binary)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # 取最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.025 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            # 轮廓是四边形
            if len(approx) == 4:
                corners = approx.reshape((4, 2))
                rect = np.zeros((4, 2), dtype="float32")
                s = corners.sum(axis=1)
                rect[0] = corners[np.argmin(s)]
                rect[2] = corners[np.argmax(s)]
                diff = np.diff(corners, axis=1)
                rect[3] = corners[np.argmax(diff)]
                rect[1] = corners[np.argmin(diff)]

                # 计算矩形中心点
                center_x = int((rect[0][0] + rect[1][0] + rect[2][0] + rect[3][0]) / 4)
                center_y = int((rect[0][1] + rect[1][1] + rect[2][1] + rect[3][1]) / 4)

                # 映射回原图坐标
                rect_center_in_ai = [center_x + crop_ai_rect[0], center_y + crop_ai_rect[1]]
                rect_center_in_original = [
                    int(rect_center_in_ai[0] * img_ai_scale[0]),
                    int(rect_center_in_ai[1] * img_ai_scale[1])
                ]

                # 判断矩形尺寸是否有效
                minW = min(rect[1][0] - rect[0][0], rect[2][0] - rect[3][0])
                minH = min(rect[3][1] - rect[0][1], rect[2][1] - rect[1][1])
                if minH > rect_min_limit and minW > rect_min_limit:
                    # 有效目标：更新卡尔曼
                    kf.update(rect_center_in_original[0], rect_center_in_original[1])
                    lost_frame_cnt = 0
                    center_to_send = rect_center_in_original
                    rect_found = True
                    last_center = rect_center_in_original
                    last_center_small = [
                        int(last_center[0] / img_ai_scale[0]),
                        int(last_center[1] / img_ai_scale[1])
                    ]
                    send_data(last_center[0]-center_pos[0], last_center[1]-center_pos[1])
                    print("目标偏差：",last_center[0]-center_pos[0], last_center[1]-center_pos[1])
                else:
                    # 矩形无效：使用卡尔曼预测
                    lost_frame_cnt += 1
                    if lost_frame_cnt > LOST_FRAME_MAX:
                        lost_frame_cnt = 0
                        kf = KalmanFilter2D(kf_init_x, kf_init_y)
                    pred_x, pred_y = kf.get_pos()
                    send_data(pred_x - center_pos[0], pred_y - center_pos[1])
            else:
                # 非四边形：使用卡尔曼预测
                lost_frame_cnt += 1
                if lost_frame_cnt > LOST_FRAME_MAX:
                    lost_frame_cnt = 0
                    kf = KalmanFilter2D(kf_init_x, kf_init_y)
                pred_x, pred_y = kf.get_pos()
                send_data(pred_x - center_pos[0], pred_y - center_pos[1])

    # ====================== 未检测到目标：卡尔曼预测 ======================
    else:
        lost_frame_cnt += 1
        if lost_frame_cnt > LOST_FRAME_MAX:
            lost_frame_cnt = 0
            kf = KalmanFilter2D(kf_init_x, kf_init_y)
        pred_x, pred_y = kf.get_pos()
        send_data(pred_x - center_pos[0], pred_y - center_pos[1])
        print("失去目标：",pred_x - center_pos[0],pred_y - center_pos[1])
        last_center = [pred_x, pred_y]

    # ====================== 画面绘制：画线 + 画点 ======================
    if hires_mode:
        # 高分辨率模式
        img.draw_line(center_pos[0], center_pos[1], last_center[0], last_center[1], image.COLOR_RED, 3)
        img.draw_circle(center_pos[0], center_pos[1], 5, image.COLOR_BLUE, -1)
        img.draw_circle(last_center[0], last_center[1], 5, image.COLOR_RED, -1)
        disp.show(img)
    else:
        # 低分辨率模式
        img_ai.draw_line(center_pos[0], center_pos[1], last_center_small[0], last_center_small[1], image.COLOR_RED, 3)
        img_ai.draw_circle(center_pos[0], center_pos[1], 3, image.COLOR_BLUE, -1)
        img_ai.draw_circle(last_center_small[0], last_center_small[1], 3, image.COLOR_RED, -1)
        disp.show(img_ai)