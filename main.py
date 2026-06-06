import cv2
import requests
import time
import os
import json
import numpy as np
import threading
from ultralytics import YOLO

# ================= CONFIG =================

RTSP_URL = "rtsp://100.87.232.62:8554/unicast"

# Paksa RTSP pakai TCP + timeout 5 detik (bukan 30 detik default)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best(1).pt")

# TELEGRAM
TELEGRAM_TOKEN = "8659715066:AAEJiPKZHJ9kFX3MfH3JanWJ21ZKzCGYqyQ"
CHAT_ID = "8060242037"

# YOLO CONFIG
CONF_THRESHOLD = 0.65
IOU_THRESHOLD = 0.5
IMG_SIZE = 640

# Minimal jumlah sampah sebelum kirim alert
TRIGGER_COUNT = 3

# Kirim maksimal tiap 1 jam
COOLDOWN = 3600

# Preview window
SHOW_PREVIEW = True
SHOW_ROI_DEBUG = True

# Class index sesuai model.names
ORGANIK_CLASSES = [0]
ANORGANIK_CLASSES = [1]

# Warna BGR untuk OpenCV
CLASS_COLORS = {
    "sampah_organik": (34, 197, 94),   # green
    "sampah_plastik": (255, 120, 0),   # blue/orange-ish in BGR
}

DEFAULT_COLORS = [
    (255, 120, 0),
    (34, 197, 94),
    (0, 255, 255),
    (255, 0, 255),
]

# ===============================================


print("🚀 Loading model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded")
print("Model classes:", model.names)


# ================= RTSP STREAM CLASS (Thread-based) =================

class RTSPStream:
    """
    Membaca frame RTSP di background thread agar main thread
    tidak pernah blocking -> window tidak 'Not Responding'.
    Auto-reconnect jika stream putus.
    """

    def __init__(self, url):
        self.url = url
        self.cap = None
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.running = False
        self._connect()
        self._start_thread()

    # ----------------------------------------------------------------
    # [DIUBAH] _connect: ditambah retry loop + validasi isOpened()
    #          dan test baca 1 frame agar tidak langsung lanjut
    #          ke _reader ketika koneksi belum benar-benar siap.
    # ----------------------------------------------------------------
    def _connect(self):
        print("📡 Connecting to stream...")

        # [BARU] Release cap lama dengan aman sebelum buat yang baru
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

        # [BARU] Retry loop — terus coba sampai berhasil atau running=False
        while self.running or self.frame is None:
            try:
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # [BARU] Validasi apakah stream benar-benar terbuka
                if self.cap.isOpened():
                    # [BARU] Test baca 1 frame untuk memastikan stream valid
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print("✅ Stream connected!")
                        with self.lock:
                            self.ret = ret
                            self.frame = frame
                        return  # Berhasil, keluar dari retry loop

                print("❌ Gagal connect / stream tidak valid, retry dalam 3 detik...")
                time.sleep(3)

            except cv2.error as e:                          # [BARU]
                print(f"❌ OpenCV error saat connect: {e}, retry dalam 3 detik...")
                time.sleep(3)
            except Exception as e:                          # [BARU]
                print(f"❌ Error saat connect: {e}, retry dalam 3 detik...")
                time.sleep(3)

    def _start_thread(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    # ----------------------------------------------------------------
    # [DIUBAH] _reader: ditambah try-except cv2.error dan Exception
    #          karena OpenCV kadang throw C++ exception langsung
    #          tanpa mengembalikan ret=False saat stream putus.
    # ----------------------------------------------------------------
    def _reader(self):
        """Thread background: terus baca frame tanpa henti."""
        while self.running:
            try:                                            # [BARU]
                ret, frame = self.cap.read()

                # [DIUBAH] tambah pengecekan frame is None
                if not ret or frame is None:
                    print("⚠️ Stream putus, reconnecting dalam 2 detik...")
                    time.sleep(2)
                    self._connect()
                    continue

                with self.lock:
                    self.ret = ret
                    self.frame = frame

            except cv2.error as e:                          # [BARU]
                # OpenCV throw C++ exception → ini penyebab error utama kamu
                print(f"⚠️ OpenCV C++ error di reader: {e}")
                print("🔄 Reconnecting dalam 2 detik...")
                time.sleep(2)
                self._connect()

            except Exception as e:                          # [BARU]
                # Tangkap semua error lain agar thread tidak mati
                print(f"⚠️ Unexpected error di reader: {e}")
                print("🔄 Reconnecting dalam 2 detik...")
                time.sleep(2)
                self._connect()

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    # ----------------------------------------------------------------
    # [DIUBAH] release: tambah try-except agar tidak error saat exit
    # ----------------------------------------------------------------
    def release(self):
        self.running = False
        try:                                                # [BARU]
            if self.cap:
                self.cap.release()
        except Exception as e:                             # [BARU]
            print(f"⚠️ Error saat release stream: {e}")


# ================= TELEGRAM FUNCTIONS =================

def encode_image(frame):
    success, img = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    if not success:
        return None

    return img.tobytes()


def send_two_photos(clean_frame, detection_report_frame, caption):
    """
    Kirim 2 gambar ke Telegram:
    1. Clean realtime image
    2. Detection report image seperti contoh:
       Segmentasi + Label | Segmentation Mask
    """

    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ Telegram token/chat_id kosong. Alert tidak dikirim.")
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMediaGroup"

        clean_img = encode_image(clean_frame)
        report_img = encode_image(detection_report_frame)

        if clean_img is None or report_img is None:
            print("❌ Gagal encode salah satu image untuk Telegram")
            return False

        media = [
            {
                "type": "photo",
                "media": "attach://clean_photo",
                "caption": caption
            },
            {
                "type": "photo",
                "media": "attach://detection_report_photo"
            }
        ]

        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "media": json.dumps(media)
            },
            files={
                "clean_photo": ("01_realtime_clean.jpg", clean_img),
                "detection_report_photo": ("02_detection_report.jpg", report_img)
            },
            timeout=20
        )

        if response.status_code == 200:
            print("✅ Telegram 2 photos sent")
            return True
        else:
            print("❌ Telegram failed:", response.text)
            return False

    except Exception as e:
        print("❌ Telegram error:", e)
        return False


# ================= ROI FUNCTION =================

def get_roi(frame):
    h, w = frame.shape[:2]

    # Sesuaikan area ROI sesuai kamera kamu
    x1 = int(w * 0.17)
    y1 = int(h * 0.10)

    x2 = int(w * 0.62)
    y2 = int(h * 0.68)

    roi = frame[y1:y2, x1:x2]

    return roi, x1, y1, x2, y2


# ================= VISUALIZATION FUNCTIONS =================

def get_class_color(class_name, cls_index):
    return CLASS_COLORS.get(class_name, DEFAULT_COLORS[cls_index % len(DEFAULT_COLORS)])


def resize_to_same_height(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    target_h = min(h1, h2)

    new_w1 = int(w1 * target_h / h1)
    new_w2 = int(w2 * target_h / h2)

    img1_resized = cv2.resize(img1, (new_w1, target_h))
    img2_resized = cv2.resize(img2, (new_w2, target_h))

    return img1_resized, img2_resized


def add_title(image, title):
    h, w = image.shape[:2]
    title_bar_h = 55

    canvas = np.zeros((h + title_bar_h, w, 3), dtype=np.uint8)
    canvas[:] = (10, 10, 20)

    canvas[title_bar_h:title_bar_h + h, 0:w] = image

    cv2.putText(
        canvas,
        title,
        (20, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return canvas


def create_segmentation_mask_image(roi, result):
    """
    Membuat panel mask seperti contoh kanan:
    background gelap + area mask per class.
    """

    h, w = roi.shape[:2]

    # Background hijau gelap agar mirip contoh
    mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    mask_canvas[:] = (20, 80, 20)

    legend_items = []

    if result.masks is not None and result.boxes is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes

        for i, mask in enumerate(masks):
            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])

            if conf < CONF_THRESHOLD:
                continue

            class_name = model.names.get(cls, str(cls))
            color = get_class_color(class_name, cls)

            # Resize mask ke ukuran ROI
            mask_resized = cv2.resize(mask, (w, h))
            binary_mask = mask_resized > 0.5

            mask_canvas[binary_mask] = color

            legend_items.append((class_name, conf, color))

    elif result.boxes is not None:
        # Fallback kalau model tidak punya masks
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < CONF_THRESHOLD:
                continue

            class_name = model.names.get(cls, str(cls))
            color = get_class_color(class_name, cls)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(mask_canvas, (x1, y1), (x2, y2), color, -1)

            legend_items.append((class_name, conf, color))

    # Legend
    legend_x = 15
    legend_y = h - 25 - (len(legend_items[:6]) * 35)

    if legend_y < 15:
        legend_y = 15

    for idx, (class_name, conf, color) in enumerate(legend_items[:6]):
        y = legend_y + idx * 35

        cv2.rectangle(mask_canvas, (legend_x, y - 18), (legend_x + 30, y + 8), color, -1)

        cv2.putText(
            mask_canvas,
            f"{class_name} ({conf:.2f})",
            (legend_x + 42, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return mask_canvas


def create_detection_report_image(roi, result):
    """
    Output 1 gambar gabungan:
    kiri  = Segmentasi + Label
    kanan = Segmentation Mask
    """

    # Panel kiri: hasil YOLO plot, sudah ada mask + label
    annotated_roi = result.plot()

    # Panel kanan: pure segmentation mask
    mask_image = create_segmentation_mask_image(roi, result)

    annotated_roi, mask_image = resize_to_same_height(annotated_roi, mask_image)

    left_panel = add_title(annotated_roi, "Segmentasi + Label")
    right_panel = add_title(mask_image, "Segmentation Mask")

    gap = 30
    h1, w1 = left_panel.shape[:2]
    h2, w2 = right_panel.shape[:2]

    final_h = max(h1, h2)
    final_w = w1 + gap + w2

    final_image = np.zeros((final_h, final_w, 3), dtype=np.uint8)
    final_image[:] = (10, 10, 20)

    final_image[0:h1, 0:w1] = left_panel
    final_image[0:h2, w1 + gap:w1 + gap + w2] = right_panel

    return final_image


def draw_summary_on_image(image, total_sampah, organik_count, organik_pct, anorganik_count, anorganik_pct):
    """
    Tambahan text summary di detection report.
    """

    output = image.copy()

    cv2.putText(
        output,
        f"Total: {total_sampah}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        f"Organik: {organik_count} ({organik_pct:.1f}%)",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        f"Anorganik: {anorganik_count} ({anorganik_pct:.1f}%)",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    return output


# ================= MAIN PROGRAM =================

# Gunakan RTSPStream (thread-based) supaya window tidak Not Responding
stream = RTSPStream(RTSP_URL)

last_sent = 0
cooldown_warning_sent = False

print("✅ YOLO sampah monitoring started...")
print("Tekan ESC untuk keluar.")
print(f"⏱️ Telegram cooldown: {COOLDOWN} detik / {COOLDOWN // 60} menit")

while True:
    ret, frame = stream.read()

    if not ret or frame is None:
        # Reconnect dihandle otomatis oleh RTSPStream._reader di background
        # Main thread tetap jalan → window tidak freeze
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    roi, x1, y1, x2, y2 = get_roi(frame)

    if roi is None or roi.size == 0:
        print("⚠️ ROI kosong. Cek koordinat ROI.")
        continue

    # Clean image untuk Telegram
    clean_frame = frame.copy()
    clean_roi = roi.copy()

    # ================= INFERENCE =================

    results = model(
        roi,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False
    )

    result = results[0]

    organik_count = 0
    anorganik_count = 0
    total_sampah = 0
    detected_items = []

    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < CONF_THRESHOLD:
                continue

            class_name = model.names.get(cls, str(cls))

            if cls in ORGANIK_CLASSES:
                organik_count += 1
                total_sampah += 1
                detected_items.append(f"{class_name} {conf:.2f}")

            elif cls in ANORGANIK_CLASSES:
                anorganik_count += 1
                total_sampah += 1
                detected_items.append(f"{class_name} {conf:.2f}")

    if total_sampah > 0:
        organik_pct = (organik_count / total_sampah) * 100
        anorganik_pct = (anorganik_count / total_sampah) * 100
    else:
        organik_pct = 0
        anorganik_pct = 0

    # ================= CREATE REPORT IMAGE =================

    detection_report = create_detection_report_image(clean_roi, result)

    detection_report = draw_summary_on_image(
        detection_report,
        total_sampah,
        organik_count,
        organik_pct,
        anorganik_count,
        anorganik_pct
    )

    # ================= ALERT LOGIC =================

    now = time.time()
    cooldown_passed = (now - last_sent) >= COOLDOWN

    if total_sampah >= TRIGGER_COUNT:
        remaining_seconds = int(COOLDOWN - (now - last_sent))

        if cooldown_passed:
            print(f"🚨 Sampah detected: {total_sampah}")

            caption = (
                f"🚨 Sampah Terdeteksi\n\n"
                f"🗑️ Total Sampah : {total_sampah}\n"
                f"🍃 Organik (Non-plastic) : {organik_count} ({organik_pct:.1f}%)\n"
                f"🧴 Anorganik (Plastic)    : {anorganik_count} ({anorganik_pct:.1f}%)\n\n"
                f"📷 Foto 1: Realtime / Clean Image\n"
                f"🔍 Foto 2: Detection Report\n\n"
                f"Detected:\n"
                f"non-plastic : {organik_count}\n"
                f"Plastic     : {anorganik_count}"
            )

            sent = send_two_photos(clean_frame, detection_report, caption)

            if sent:
                last_sent = now
                cooldown_warning_sent = False
                print("✅ Cooldown dimulai. Alert berikutnya maksimal 1 jam lagi.")

        else:
            # [DIUBAH] hanya tampil sekali saat sisa waktu <= 10 menit
            if remaining_seconds <= 600 and not cooldown_warning_sent:
                cooldown_warning_sent = True  # [BARU] flag agar hanya sekali
                print(
                    f"⏳ Sampah terdeteksi {total_sampah}, "
                    f"tapi cooldown belum selesai. "
                    f"Sisa {remaining_seconds} detik."
                )

    # ================= DISPLAY =================

    if SHOW_PREVIEW:
        if SHOW_ROI_DEBUG:
            debug_frame = frame.copy()

            cv2.rectangle(
                debug_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

            cv2.putText(
                debug_frame,
                "Detection ROI",
                (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.imshow("Original Frame + ROI", debug_frame)

        cv2.imshow("Detection Report", detection_report)

        # waitKey(1) wajib dipanggil tiap loop agar window tetap responsif
        if cv2.waitKey(1) & 0xFF == 27:
            break

stream.release()
cv2.destroyAllWindows()
print("🛑 Monitoring stopped.")