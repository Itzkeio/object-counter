import cv2
import numpy as np
from ultralytics import YOLO

# ====== Sumber video ======
cap = cv2.VideoCapture("people2.mp4")   # ganti ke 0 untuk webcam
assert cap.isOpened(), "Gagal membuka sumber video."

# ====== KONFIGURASI ======
MODEL_PATH = "person_v3.pt"
CONF_TH = 0.25
IOU_TH = 0.4
TRACKER_CFG = "botsort.yaml"
PERSON_CLASS_ID = 0
STATE_DEBOUNCE_FRAMES = 2

# ====== GARIS ROI ======
LINE_IN_1 = [(760, 77), (189, 932)]      # garis masuk utama
LINE_OUT = [(794, 167), (272, 1072)]     # garis keluar
LINE_IN_2 = [(1088, 1), (1084, 1066)]    # backup masuk jika tidak lewat IN1

# ====== UTIL ======
def put_text(img, text, org, scale=0.8, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness, cv2.LINE_AA)

def line_side(pt, line_p1, line_p2):
    """Mengembalikan nilai sisi relatif terhadap garis (positif/negatif)"""
    x, y = pt
    x1, y1 = line_p1
    x2, y2 = line_p2
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

def draw_lines(img):
    cv2.line(img, LINE_IN_1[0], LINE_IN_1[1], (0, 255, 0), 3)   # hijau = IN utama
    cv2.line(img, LINE_OUT[0], LINE_OUT[1], (0, 0, 255), 3)     # merah = OUT
    cv2.line(img, LINE_IN_2[0], LINE_IN_2[1], (0, 200, 255), 2) # kuning = backup IN2

# ====== COUNTER ======
class DirectionalLineCounter:
    def __init__(self, debounce_frames=2):
        self.debounce = debounce_frames
        self.count_in = 0
        self.count_out = 0
        self.last_pos = {}          # track_id -> prev_point
        self.last_switch = {}       # track_id -> frame index
        self.in_counted = set()     # ID yang sudah dihitung masuk
        self.out_counted = set()    # ID yang sudah dihitung keluar

    def reset(self):
        self.count_in = 0
        self.count_out = 0
        self.last_pos.clear()
        self.last_switch.clear()
        self.in_counted.clear()
        self.out_counted.clear()

    def update(self, tid, cx, cy, frame_idx):
        cur_pt = (cx, cy)
        if tid not in self.last_pos:
            self.last_pos[tid] = cur_pt
            self.last_switch[tid] = frame_idx
            return

        prev_pt = self.last_pos[tid]
        if frame_idx - self.last_switch[tid] < self.debounce:
            return

        prev_side_in1 = line_side(prev_pt, *LINE_IN_1)
        curr_side_in1 = line_side(cur_pt, *LINE_IN_1)
        prev_side_in2 = line_side(prev_pt, *LINE_IN_2)
        curr_side_in2 = line_side(cur_pt, *LINE_IN_2)
        prev_side_out = line_side(prev_pt, *LINE_OUT)
        curr_side_out = line_side(cur_pt, *LINE_OUT)

        # ==== DETEKSI IN (garis utama) ====
        if tid not in self.in_counted and (prev_side_in1 < 0 and curr_side_in1 > 0):
            self.count_in += 1
            self.in_counted.add(tid)
            self.last_switch[tid] = frame_idx

        # ==== DETEKSI IN2 (backup, hanya aktif jika belum lewat IN1) ====
        elif tid not in self.in_counted and (prev_side_in2 < 0 and curr_side_in2 > 0):
            self.count_in += 1
            self.in_counted.add(tid)
            self.last_switch[tid] = frame_idx

        # ==== DETEKSI OUT (arah berlawanan) ====
        elif tid not in self.out_counted and (prev_side_out > 0 and curr_side_out < 0):
            self.count_out += 1
            self.out_counted.add(tid)
            self.last_switch[tid] = frame_idx

        self.last_pos[tid] = cur_pt


def main():
    model = YOLO(MODEL_PATH)
    counter = DirectionalLineCounter(debounce_frames=STATE_DEBOUNCE_FRAMES)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Selesai.")
            break

        # ====== DETEKSI + TRACKING ======
        results = model.track(
            frame,
            conf=CONF_TH,
            iou=IOU_TH,
            classes=[PERSON_CLASS_ID],
            persist=True,
            tracker=TRACKER_CFG,
            verbose=False
        )

        vis = frame.copy()
        draw_lines(vis)

        if results and len(results) > 0:
            r = results[0]
            boxes = r.boxes
            if boxes is not None and boxes.id is not None:
                ids   = boxes.id.cpu().numpy().astype(int)
                xyxy  = boxes.xyxy.cpu().numpy()
                cls   = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                for i, tid in enumerate(ids):
                    if cls[i] != PERSON_CLASS_ID:
                        continue

                    x1, y1, x2, y2 = xyxy[i]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # Update counter
                    counter.update(tid, cx, cy, frame_idx)
                    # print(tid)

                    # Visualisasi
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (50, 220, 50), 2)
                    cv2.circle(vis, (cx, cy), 4, (255, 255, 255), -1)
                    put_text(vis, f"ID {tid} ({confs[i]:.2f})", (int(x1), int(y1) - 8), 0.55, 1)

        # ====== HUD ======
        put_text(vis, f"IN: {counter.count_in}   OUT: {counter.count_out}", (15, 40), 0.95, 2)
        put_text(vis, "Metode: Line Crossing", (15, 80), 0.6, 1)
        put_text(vis, "r: reset  q: quit", (15, 112), 0.6, 1)

        cv2.imshow("People Counter (Directional In/Out)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            counter.reset()

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"IN total = {counter.count_in}")
    print(f"OUT total = {counter.count_out}")

if __name__ == "__main__":
    main()
