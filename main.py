from maix import camera, display, image, nn, app, uart, touchscreen
import cv2
import numpy as np
from struct import pack
import time

#虚拟按键程序
class VirtualButtons:
    def __init__(self):
        self.buttons = [
            [80, 30, 100, 40, "Apriltag", "apri"],
            [260, 30, 100, 40, "Circle", "circle"],
            [80, 330, 100, 40, "ID:6", "id:6"],
            [260, 330, 100, 40, "ID:59", "id:59"]
        ]
        self.touch_areas = [
            [70, 20, 120, 60],
            [250, 20, 120, 60],
            [70, 280, 120, 60],
            [250, 280, 120, 60]
        ]
        self.last_touch_time = 0
        self.touch_debounce = 0.1

    def check_touch(self, touch_x, touch_y):
        current_time = time.time()
        if current_time - self.last_touch_time < self.touch_debounce:
            return None
        tx = touch_x
        ty = touch_y
        for i, touch_area in enumerate(self.touch_areas):
            ax, ay, aw, ah = touch_area
            if ax <= tx <= ax + aw and ay <= ty <= ay + ah:
                self.last_touch_time = current_time
                return self.buttons[i][5]
        return None

    def draw_buttons(self, img, current_mode, selected_id):
        img_cv = image.image2cv(img)
        for button in self.buttons:
            x, y, w, h, text, action = button
            highlight = False
            if (action == "apri" and current_mode == "apri") or (action == "circle" and current_mode == "circle"):
                highlight = True
            if action == f"id:{selected_id}":
                highlight = True
            if highlight:
                color = (0, 255, 255)
                thickness = 3
            else:
                color = (255, 0, 255)
                thickness = 2
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, thickness)
            text_x = x + 10
            text_y = y + 25
            cv2.putText(img_cv, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return image.cv2image(img_cv)


def init_touchscreen():
    try:
        ts = touchscreen.TouchScreen()
        return ts
    except Exception as e:
        return None


class KalmanFilter2D:
    def __init__(self, init_x, init_y):
        self.state = np.array([[init_x], [0.0], [init_y], [0.0]], dtype=np.float32)
        self.F = np.array([[1, 1, 0, 0],[0, 1, 0, 0],[0, 0, 1, 1],[0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],[0, 0, 1, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 0.072
        self.R = np.eye(2, dtype=np.float32) * 23.0
        self.P = np.eye(4, dtype=np.float32) * 10.0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, meas_x, meas_y):
        z = np.array([[meas_x], [meas_y]], dtype=np.float32)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_pos(self):
        return int(self.state[0, 0]), int(self.state[2, 0])


uart_dev = uart.UART("/dev/ttyS0", 115200)

def send_data(err_x, err_y):
    err_x_h = (err_x >> 8) & 0xff
    err_x_l = err_x & 0xff
    err_y_h = (err_y >> 8) & 0xff
    err_y_l = err_y & 0xff
    data = pack("<BBBBBB", 0xAA, err_x_h, err_x_l, err_y_h, err_y_l, 0x5C)
    uart_dev.write(data)


LOST_FRAME_MAX = 50
lost_frame_cnt = 0
kf_init_x = 253
kf_init_y = 211
kf = KalmanFilter2D(kf_init_x, kf_init_y)

crop_padding = 23
rect_min_limit = 12
std_from_white_rect = True
hires_mode = True
high_res = 448
model_path = "/root/models/model_3356.mud"
cam_buff_num = 2

detector = nn.YOLOv5(model=model_path, dual_buff=True)
if hires_mode:
    cam = camera.Camera(high_res, high_res, detector.input_format(), buff_num=cam_buff_num)
else:
    cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format(), buff_num=cam_buff_num)

disp = display.Display()
ts = init_touchscreen()
buttons = VirtualButtons()
current_mode = "circle"
selected_id = -1

families = image.ApriltagFamilies.TAG36H11
x_scale = cam.width() / 160
y_scale = cam.height() / 120

center_pos = [253, 211]
last_center = center_pos
last_center_small = [detector.input_width(), detector.input_height()]

while not app.need_exit():
    rect_found = False
    img = cam.read()

    if ts:
        try:
            x, y, pressed = ts.read()
            #print(x,y)
            if pressed:
                action = buttons.check_touch(x, y)
                if action == "apri":
                    current_mode = "apri"
                    selected_id = -1
                elif action == "circle":
                    current_mode = "circle"
                    selected_id = -1
                elif action == "id:6":
                    selected_id = 6
                    current_mode = "apri"
                elif action == "id:59":
                    selected_id = 59
                    current_mode = "apri"
        except:
            pass

    if current_mode == "apri":
        new_img = img.resize(160, 120)
        apriltags = new_img.find_apriltags(families=families)
        target_found = False
        if apriltags:
            for a in apriltags:
                tag_id = a.id()
                if selected_id != -1 and tag_id != selected_id:
                    continue

                corners = a.corners()
                for i in range(4):
                    corners[i][0] = int(corners[i][0] * x_scale)
                    corners[i][1] = int(corners[i][1] * y_scale)

                cx = int((corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4)
                cy = int((corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4)

                # for i in range(4):
                #     img.draw_line(corners[i][0], corners[i][1], corners[(i+1)%4][0], corners[(i+1)%4][1], image.COLOR_RED, 2)
                img.draw_circle(cx, cy, 10, image.COLOR_GREEN, -1)
                img.draw_string(cx + 20, cy, f"ID:{tag_id}", image.COLOR_GREEN)
                
                last_center = [cx, cy]
                send_data(cx - center_pos[0], cy - center_pos[1])
                target_found = True
        if not target_found:
            send_data(0, 0)

    else:
        if hires_mode:
            img_ai = img.resize(detector.input_width(), detector.input_height())
        else:
            img_ai = img

        objs = detector.detect(img_ai, conf_th=0.25, iou_th=0.45)
        max_idx = -1
        max_s = 0
        for i, obj in enumerate(objs):
            s = obj.w * obj.h
            if s > max_s:
                max_s = s
                max_idx = i

        kf.predict()
        center_to_send = [0, 0]

        if max_idx >= 0:
            obj = objs[max_idx]
            w = obj.w + crop_padding * 2
            h = obj.h + crop_padding * 2
            w = w + 1 if w % 2 != 0 else w
            h = h + 1 if h % 2 != 0 else h
            x = obj.x - crop_padding
            y = obj.y - crop_padding
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

            crop_ai = img_ai.crop(x, y, w, h)
            crop_ai_rect = [x, y, w, h]
            img_ai_scale = [img.width() / img_ai.width(), img.height() / img_ai.height()]
            gray = crop_ai.to_format(image.Format.FMT_GRAYSCALE)
            gray_cv = image.image2cv(gray, False, False)
            binary = cv2.adaptiveThreshold(gray_cv, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 23, 16)

            if std_from_white_rect:
                h_bin, w_bin = binary.shape[:2]
                mask = np.zeros((h_bin + 2, w_bin + 2), np.uint8)
                cv2.floodFill(binary, mask, (2, 2), 255, loDiff=15, upDiff=15, flags=4)
                cv2.floodFill(binary, mask, (w_bin - 2, h_bin - 2), 255, loDiff=15, upDiff=15, flags=4)
                binary = cv2.bitwise_not(binary)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.025 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                if len(approx) == 4:
                    corners = approx.reshape((4, 2))
                    rect = np.zeros((4, 2), dtype="float32")
                    s = corners.sum(axis=1)
                    rect[0] = corners[np.argmin(s)]
                    rect[2] = corners[np.argmax(s)]
                    diff = np.diff(corners, axis=1)
                    rect[3] = corners[np.argmax(diff)]
                    rect[1] = corners[np.argmin(diff)]
                    center_x = int((rect[0][0] + rect[1][0] + rect[2][0] + rect[3][0]) / 4)
                    center_y = int((rect[0][1] + rect[1][1] + rect[2][1] + rect[3][1]) / 4)
                    rect_center_in_ai = [center_x + crop_ai_rect[0], center_y + crop_ai_rect[1]]
                    rect_center_in_original = [
                        int(rect_center_in_ai[0] * img_ai_scale[0]),
                        int(rect_center_in_ai[1] * img_ai_scale[1])
                    ]
                    minW = min(rect[1][0] - rect[0][0], rect[2][0] - rect[3][0])
                    minH = min(rect[3][1] - rect[0][1], rect[2][1] - rect[1][1])
                    if minH > rect_min_limit and minW > rect_min_limit:
                        kf.update(rect_center_in_original[0], rect_center_in_original[1])
                        lost_frame_cnt = 0
                        center_to_send = rect_center_in_original
                        rect_found = True
                        last_center = rect_center_in_original
                        send_data(last_center[0]-center_pos[0], last_center[1]-center_pos[1])
                    else:
                        lost_frame_cnt += 1
                        if lost_frame_cnt > LOST_FRAME_MAX:
                            lost_frame_cnt = 0
                            kf = KalmanFilter2D(kf_init_x, kf_init_y)
                        pred_x, pred_y = kf.get_pos()
                        send_data(pred_x - center_pos[0], pred_y - center_pos[1])
                else:
                    lost_frame_cnt += 1
                    if lost_frame_cnt > LOST_FRAME_MAX:
                        lost_frame_cnt = 0
                        kf = KalmanFilter2D(kf_init_x, kf_init_y)
                    pred_x, pred_y = kf.get_pos()
                    send_data(pred_x - center_pos[0], pred_y - center_pos[1])
        else:
            lost_frame_cnt += 1
            if lost_frame_cnt > LOST_FRAME_MAX:
                lost_frame_cnt = 0
                kf = KalmanFilter2D(kf_init_x, kf_init_y)
            pred_x, pred_y = kf.get_pos()
            send_data(pred_x - center_pos[0], pred_y - center_pos[1])
            last_center = [pred_x, pred_y]

    img.draw_circle(center_pos[0], center_pos[1], 5, image.COLOR_BLUE, -1)
    if current_mode == "circle":
        img.draw_line(center_pos[0], center_pos[1], last_center[0], last_center[1], image.COLOR_RED, 3)
        img.draw_circle(last_center[0], last_center[1], 5, image.COLOR_RED, -1)

    img = buttons.draw_buttons(img, current_mode, selected_id)
    disp.show(img)