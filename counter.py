import cv2
import torch
import concurrent.futures
import os
from ultralytics import YOLO

# ====== KONFIGURASI INPUT ======
VIDEO_SOURCE = "people.mp4"

# ====== KONFIGURASI MODEL ======
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

# ====== KONFIGURASI ANTI-DUPLICATE ======
# Jika sebuah track yang sudah dihitung berada dalam radius (px) dan dalam
# frame window (frame) dari posisi sekarang, anggap itu orang yang sama.
COUNT_PROXIMITY = 80       # pixels radius untuk menganggap 'sama'
COUNT_FRAME_WINDOW = 50    # frames; jendela waktu untuk proximity check


# ====== UTIL ======
def put_text(img, text, org, scale=0.8, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness, cv2.LINE_AA)


def line_side(pt, line_p1, line_p2):
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
    def __init__(self, debounce_frames=2, proximity=COUNT_PROXIMITY, frame_window=COUNT_FRAME_WINDOW):
        self.debounce = debounce_frames
        self.count_in = 0
        self.count_out = 0
        self.last_pos = {}      # tid -> (x,y)
        self.last_switch = {}   # tid -> frame_idx when last switched / counted
        self.in_counted = set() # set of tids that already counted as "in"
        self.out_counted = set()
        # tambahan untuk anti-duplicate ketika tracker reassign ID
        self.proximity = proximity
        self.frame_window = frame_window

    def reset(self):
        self.__init__(self.debounce, self.proximity, self.frame_window)

    def _recently_counted_near(self, cx, cy, counted_set, frame_idx):
        """
        Cek apakah ada tid di counted_set yang posisinya 'dekat' (euclidean) dengan (cx,cy)
        dan juga dihitung dalam frame window terakhir. Jika ya, anggap sebagai duplikat.
        """
        px = cx; py = cy
        for counted_tid in counted_set:
            if counted_tid in self.last_pos and counted_tid in self.last_switch:
                lx, ly = self.last_pos[counted_tid]
                last_frame = self.last_switch[counted_tid]
                # hanya pertimbangkan yang baru saja dihitung (dalam frame_window)
                if frame_idx - last_frame <= self.frame_window:
                    dist2 = (lx - px) ** 2 + (ly - py) ** 2
                    if dist2 <= (self.proximity ** 2):
                        return True
        return False

    def update(self, tid, cx, cy, frame_idx):
        cur_pt = (cx, cy)
        # inisialisasi posisi baru
        if tid not in self.last_pos:
            self.last_pos[tid] = cur_pt
            self.last_switch[tid] = frame_idx
            return

        prev_pt = self.last_pos[tid]

        # debounce simple berdasarkan last_switch
        if frame_idx - self.last_switch[tid] < self.debounce:
            # update posisi tapi jangan hitung
            self.last_pos[tid] = cur_pt
            return

        prev_in1 = line_side(prev_pt, *LINE_IN_1)
        curr_in1 = line_side(cur_pt, *LINE_IN_1)
        prev_in2 = line_side(prev_pt, *LINE_IN_2)
        curr_in2 = line_side(cur_pt, *LINE_IN_2)
        prev_out = line_side(prev_pt, *LINE_OUT)
        curr_out = line_side(cur_pt, *LINE_OUT)

        # ---- LOGIKA HITUNG DENGAN ANTI-DUPLICATE ----
        # IN via LINE_IN_1
        if (prev_in1 < 0 and curr_in1 > 0):
            # jika tid sudah dihitung sebagai IN sebelumnya -> skip
            if tid in self.in_counted:
                pass
            else:
                # cek apakah ada counted IN lain yang baru saja dihitung dan dekat -> duplikat
                if self._recently_counted_near(cx, cy, self.in_counted, frame_idx):
                    # treat as duplicate: skip counting
                    pass
                else:
                    self.count_in += 1
                    self.in_counted.add(tid)
                    self.last_switch[tid] = frame_idx

        # IN via LINE_IN_2 (backup)
        elif (prev_in2 < 0 and curr_in2 > 0):
            if tid in self.in_counted:
                pass
            else:
                if self._recently_counted_near(cx, cy, self.in_counted, frame_idx):
                    pass
                else:
                    self.count_in += 1
                    self.in_counted.add(tid)
                    self.last_switch[tid] = frame_idx

        # OUT via LINE_OUT
        elif (prev_out > 0 and curr_out < 0):
            if tid in self.out_counted:
                pass
            else:
                if self._recently_counted_near(cx, cy, self.out_counted, frame_idx):
                    pass
                else:
                    self.count_out += 1
                    self.out_counted.add(tid)
                    self.last_switch[tid] = frame_idx

        # update posisi terakhir
        self.last_pos[tid] = cur_pt


# ====== FUNGSI DETEKSI SUMBER VIDEO ======
def open_video_source(source):
    """Deteksi apakah input RTSP stream atau file lokal"""
    if isinstance(source, str) and source.startswith("rtsp://"):
        print("📡 Menggunakan sumber RTSP stream...")
        cap = cv2.VideoCapture(f"{source}?rtsp_transport=tcp", cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif os.path.isfile(source):
        print("🎞️ Menggunakan file video lokal...")
        cap = cv2.VideoCapture(source)
    else:
        print("⚠️ Sumber video tidak valid!")
        return None

    if not cap.isOpened():
        print("❌ Gagal membuka sumber video.")
        return None
    return cap


# ====== MAIN ======
def main():
    print(f"🧠 Loading model on {'GPU' if torch.cuda.is_available() else 'CPU'}...")
    model = YOLO(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    counter = DirectionalLineCounter(STATE_DEBOUNCE_FRAMES)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    cap = open_video_source(VIDEO_SOURCE)
    if cap is None:
        return

    frame_idx = 0
    future = None
    results = None

    print("✅ Stream connected. Tekan [q] untuk keluar.")

    # Deteksi FPS hanya jika video lokal
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30  # fallback default untuk RTSP atau jika FPS tidak terbaca
    delay = int(1000 / fps)  # konversi ke milidetik untuk waitKey

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️ Stream terputus atau video selesai.")
            break

        # Jalankan inference di thread terpisah
        if future is None or future.done():
            if future is not None:
                results = future.result()
            future = executor.submit(
                lambda: model.track(
                    frame,
                    conf=CONF_TH,
                    iou=IOU_TH,
                    classes=[PERSON_CLASS_ID],
                    persist=True,
                    tracker=TRACKER_CFG,
                    verbose=False
                )
            )

        vis = frame.copy()
        draw_lines(vis)

        if results and len(results) > 0:
            r = results[0]
            boxes = r.boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                for i, tid in enumerate(ids):
                    if cls[i] != PERSON_CLASS_ID:
                        continue
                    x1, y1, x2, y2 = xyxy[i]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    counter.update(tid, cx, cy, frame_idx)

                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (50, 220, 50), 2)
                    cv2.circle(vis, (cx, cy), 4, (255, 255, 255), -1)
                    put_text(vis, f"ID {tid} ({confs[i]:.2f})", (int(x1), int(y1) - 8), 0.55, 1)

        put_text(vis, f"IN: {counter.count_in}   OUT: {counter.count_out}", (15, 40), 0.95, 2)
        put_text(vis, "Metode: Line Crossing", (15, 80), 0.6, 1)
        put_text(vis, "r: reset  q: quit", (15, 112), 0.6, 1)

        cv2.imshow("People Counter (Video/RTSP)", vis)
        key = cv2.waitKey(delay) & 0xFF  # gunakan delay sesuai fps
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
