import cv2
import requests
import time
import os
import json
import numpy as np
import threading
from ultralytics import YOLO


# =========================================================
# CONFIGURATION
# =========================================================

RTSP_URL = "rtsp://100.87.232.62:8554/unicast"

# Memaksa OpenCV menggunakan RTSP TCP dan timeout 5 detik.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|stimeout;5000000"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best(1).pt")


# =========================================================
# TELEGRAM CONFIGURATION
# =========================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8659715066:AAEJiPKZHJ9kFX3MfH3JanWJ21ZKzCGYqyQ")
CHAT_IDS = os.getenv("TELEGRAM_CHAT_IDS", "8060242037,8416897843").split(",")

LOCATION_NAME = "Sungai Dam Oongan"

MAP_URL = (
    "https://www.google.com/maps/place/Sungai+Dam+Oongan/"
    "@-8.6329877,115.2334963,3a,75y,90h,75t/data=!3m7!1e1!3m5!"
    "1s5aJu55ZaBFGcmsLOE9A06g!2e0!6shttps:%2F%2Fstreetviewpixels-pa."
    "googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile"
    "%26w%3D900%26h%3D600%26pitch%3D15.00474726809989"
    "%26panoid%3D5aJu55ZaBFGcmsLOE9A06g%26yaw%3D90!"
    "7i16384!8i8192!4m15!1m8!3m7!"
    "1s0x2dd23f75d290b29b:0xafda2f5449a759f5!"
    "2sSungai+Dam+Oongan!8m2!3d-8.6330075!4d115.2334932!"
    "10e5!16s%2Fg%2F11c0xcywl8!3m5!"
    "1s0x2dd23f75d290b29b:0xafda2f5449a759f5!"
    "8m2!3d-8.6330075!4d115.2334932!"
    "16s%2Fg%2F11c0xcywl8?entry=ttu"
)


# =========================================================
# YOLO CONFIGURATION
# =========================================================

CONF_THRESHOLD = 0.65
IOU_THRESHOLD = 0.5
IMG_SIZE = 640

# Alert dikirim jika jumlah sampah minimal mencapai nilai ini.
TRIGGER_COUNT = 3

# Jeda alert Telegram dalam detik.
# 3600 detik = 1 jam.
COOLDOWN = 3600

SHOW_PREVIEW = True
SHOW_ROI_DEBUG = True

# Sesuaikan dengan model.names.
ORGANIK_CLASSES = [0]
ANORGANIK_CLASSES = [1]

CLASS_COLORS = {
    "sampah_organik": (34, 197, 94),
    "sampah_plastik": (255, 120, 0),
}

DEFAULT_COLORS = [
    (255, 120, 0),
    (34, 197, 94),
    (0, 255, 255),
    (255, 0, 255),
]


# =========================================================
# LOAD MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model tidak ditemukan:\n{MODEL_PATH}\n\n"
        "Pastikan file best(1).pt berada satu folder dengan script Python."
    )

print("🚀 Loading model...")

model = YOLO(MODEL_PATH)

print("✅ Model loaded")
print("Model classes:", model.names)


# =========================================================
# RTSP STREAM
# =========================================================

class RTSPStream:
    """
    Membaca RTSP stream pada background thread.

    Tujuannya agar proses utama tidak freeze dan OpenCV window
    tidak menjadi Not Responding ketika stream terputus.
    """

    def __init__(self, url):
        self.url = url
        self.cap = None
        self.frame = None
        self.ret = False

        self.lock = threading.Lock()
        self.connection_lock = threading.Lock()

        self.running = True

        self._connect()

        self.thread = threading.Thread(
            target=self._reader,
            daemon=True
        )

        self.thread.start()

    def _connect(self):
        """
        Membuka koneksi RTSP dengan retry otomatis.
        """

        with self.connection_lock:
            print("📡 Connecting to stream...")

            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass

            while self.running:
                try:
                    cap = cv2.VideoCapture(
                        self.url,
                        cv2.CAP_FFMPEG
                    )

                    cap.set(
                        cv2.CAP_PROP_BUFFERSIZE,
                        1
                    )

                    if cap.isOpened():
                        ret, frame = cap.read()

                        if ret and frame is not None:
                            self.cap = cap

                            with self.lock:
                                self.ret = True
                                self.frame = frame

                            print("✅ Stream connected!")
                            return True

                    cap.release()

                    print(
                        "❌ Stream tidak valid. "
                        "Retry dalam 3 detik..."
                    )

                    time.sleep(3)

                except cv2.error as error:
                    print(
                        f"❌ OpenCV error saat connect: {error}"
                    )

                    time.sleep(3)

                except Exception as error:
                    print(
                        f"❌ Error saat connect: {error}"
                    )

                    time.sleep(3)

        return False

    def _reader(self):
        """
        Membaca frame terus-menerus pada background thread.
        """

        while self.running:
            try:
                if self.cap is None:
                    self._connect()
                    continue

                ret, frame = self.cap.read()

                if not ret or frame is None:
                    print(
                        "⚠️ Stream putus. "
                        "Reconnect dalam 2 detik..."
                    )

                    with self.lock:
                        self.ret = False

                    time.sleep(2)
                    self._connect()
                    continue

                with self.lock:
                    self.ret = True
                    self.frame = frame

            except cv2.error as error:
                print(
                    f"⚠️ OpenCV error pada RTSP reader: {error}"
                )

                with self.lock:
                    self.ret = False

                time.sleep(2)
                self._connect()

            except Exception as error:
                print(
                    f"⚠️ Unexpected RTSP error: {error}"
                )

                with self.lock:
                    self.ret = False

                time.sleep(2)
                self._connect()

    def read(self):
        """
        Mengambil frame terbaru.
        """

        with self.lock:
            if self.frame is None:
                return False, None

            return self.ret, self.frame.copy()

    def release(self):
        """
        Menghentikan stream dengan aman.
        """

        self.running = False

        try:
            if self.cap is not None:
                self.cap.release()
        except Exception as error:
            print(
                f"⚠️ Error saat release stream: {error}"
            )


# =========================================================
# TELEGRAM FUNCTIONS
# =========================================================

def encode_image(frame):
    """
    Mengubah OpenCV frame menjadi JPEG bytes.
    """

    success, image = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    if not success:
        return None

    return image.tobytes()


def send_two_photos(
    clean_frame,
    detection_report_frame,
    caption
):
    """
    Mengirim dua foto ke Telegram:

    1. Foto kamera asli.
    2. Detection report yang berisi label dan segmentation mask.
    """

    if not TELEGRAM_TOKEN:
        print(
            "⚠️ TELEGRAM_TOKEN belum diatur."
        )
        return False

    if not CHAT_IDS:
        print(
            "⚠️ TELEGRAM_CHAT_ID belum diatur."
        )
        return False

    try:
        clean_image = encode_image(clean_frame)
        report_image = encode_image(detection_report_frame)

        if clean_image is None:
            print("❌ Gagal encode clean image.")
            return False

        if report_image is None:
            print("❌ Gagal encode detection report.")
            return False

        telegram_url = (
            f"https://api.telegram.org/"
            f"bot{TELEGRAM_TOKEN}/sendMediaGroup"
        )

        media = [
            {
                "type": "photo",
                "media": "attach://clean_photo",
                "caption": caption,
                "parse_mode": "HTML"
            },
            {
                "type": "photo",
                "media": "attach://detection_report_photo"
            }
        ]

        success_count = 0

        for chat_id in CHAT_IDS:  # ✅ loop ke setiap Chat ID
            response = requests.post(
                telegram_url,
                data={
                    "chat_id": chat_id.strip(),
                    "media": json.dumps(media)
                },
                files={
                    "clean_photo": (
                        "01_realtime_clean.jpg",
                        clean_image,
                        "image/jpeg"
                    ),
                    "detection_report_photo": (
                        "02_detection_report.jpg",
                        report_image,
                        "image/jpeg"
                    )
                },
                timeout=30
            )

            if response.status_code == 200:
                print(f"✅ Foto terkirim ke chat_id: {chat_id.strip()}")
                success_count += 1
            else:
                print(f"❌ Gagal kirim ke {chat_id.strip()}:", response.status_code, response.text)

        return success_count > 0

    except requests.RequestException as error:
        print(f"❌ Telegram network error: {error}")
        return False

    except Exception as error:
        print(f"❌ Telegram unexpected error: {error}")
        return False

    except requests.RequestException as error:
        print(
            f"❌ Telegram network error: {error}"
        )
        return False

    except Exception as error:
        print(
            f"❌ Telegram unexpected error: {error}"
        )
        return False


# =========================================================
# REGION OF INTEREST
# =========================================================

def get_roi(frame):
    """
    Mengambil area yang ingin diproses YOLO.

    Nilai menggunakan persentase frame sehingga tetap relatif
    terhadap resolusi kamera.
    """

    height, width = frame.shape[:2]

    x1 = int(width * 0.17)
    y1 = int(height * 0.10)

    x2 = int(width * 0.62)
    y2 = int(height * 0.68)

    roi = frame[y1:y2, x1:x2]

    return roi, x1, y1, x2, y2


# =========================================================
# VISUALIZATION FUNCTIONS
# =========================================================

def get_model_class_name(class_index):
    """
    Mengambil nama class dari model YOLO.
    """

    if isinstance(model.names, dict):
        return model.names.get(
            class_index,
            str(class_index)
        )

    if (
        isinstance(model.names, list)
        and class_index < len(model.names)
    ):
        return model.names[class_index]

    return str(class_index)


def get_class_color(
    class_name,
    class_index
):
    """
    Mendapatkan warna berdasarkan class.
    """

    return CLASS_COLORS.get(
        class_name,
        DEFAULT_COLORS[
            class_index % len(DEFAULT_COLORS)
        ]
    )


def resize_to_same_height(
    image_left,
    image_right
):
    """
    Menyamakan tinggi dua gambar sebelum digabung.
    """

    left_height, left_width = image_left.shape[:2]
    right_height, right_width = image_right.shape[:2]

    target_height = min(
        left_height,
        right_height
    )

    new_left_width = int(
        left_width * target_height / left_height
    )

    new_right_width = int(
        right_width * target_height / right_height
    )

    resized_left = cv2.resize(
        image_left,
        (new_left_width, target_height)
    )

    resized_right = cv2.resize(
        image_right,
        (new_right_width, target_height)
    )

    return resized_left, resized_right


def add_title(
    image,
    title
):
    """
    Menambahkan title bar di atas panel.
    """

    height, width = image.shape[:2]
    title_bar_height = 55

    canvas = np.zeros(
        (
            height + title_bar_height,
            width,
            3
        ),
        dtype=np.uint8
    )

    canvas[:] = (10, 10, 20)

    canvas[
        title_bar_height:
        title_bar_height + height,
        0:width
    ] = image

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


# def create_segmentation_mask_image(
#     roi,
#     result
# ):
#     """
#     Membuat gambar segmentation mask.

#     Jika model merupakan segmentation model, mask asli
#     digunakan. Jika model hanya object detection, bounding box
#     digunakan sebagai fallback.
#     """

#     height, width = roi.shape[:2]

#     mask_canvas = np.zeros(
#         (height, width, 3),
#         dtype=np.uint8
#     )

#     mask_canvas[:] = (20, 80, 20)

#     legend_items = []

#     if (
#         result.masks is not None
#         and result.boxes is not None
#     ):
#         masks = (
#             result.masks.data
#             .cpu()
#             .numpy()
#         )

#         boxes = result.boxes

#         object_count = min(
#             len(masks),
#             len(boxes)
#         )

#         for index in range(object_count):
#             class_index = int(
#                 boxes.cls[index].item()
#             )

#             confidence = float(
#                 boxes.conf[index].item()
#             )

#             if confidence < CONF_THRESHOLD:
#                 continue

#             class_name = get_model_class_name(
#                 class_index
#             )

#             color = get_class_color(
#                 class_name,
#                 class_index
#             )

#             resized_mask = cv2.resize(
#                 masks[index],
#                 (width, height),
#                 interpolation=cv2.INTER_NEAREST
#             )

#             binary_mask = resized_mask > 0.5

#             mask_canvas[binary_mask] = color

#             legend_items.append(
#                 (
#                     class_name,
#                     confidence,
#                     color
#                 )
#             )

#     elif result.boxes is not None:
#         for box in result.boxes:
#             class_index = int(
#                 box.cls[0].item()
#             )

#             confidence = float(
#                 box.conf[0].item()
#             )

#             if confidence < CONF_THRESHOLD:
#                 continue

#             class_name = get_model_class_name(
#                 class_index
#             )

#             color = get_class_color(
#                 class_name,
#                 class_index
#             )

#             box_coordinates = (
#                 box.xyxy[0]
#                 .cpu()
#                 .numpy()
#                 .astype(int)
#             )

#             box_x1, box_y1, box_x2, box_y2 = (
#                 box_coordinates
#             )

#             cv2.rectangle(
#                 mask_canvas,
#                 (box_x1, box_y1),
#                 (box_x2, box_y2),
#                 color,
#                 -1
#             )

#             legend_items.append(
#                 (
#                     class_name,
#                     confidence,
#                     color
#                 )
#             )

#     maximum_legend_items = 8
#     visible_legend_items = legend_items[
#         :maximum_legend_items
#     ]

#     legend_x = 20
#     line_height = 34

#     legend_height = (
#         len(visible_legend_items)
#         * line_height
#     )

#     legend_y = max(
#         30,
#         height - legend_height - 20
#     )

#     for index, item in enumerate(
#         visible_legend_items
#     ):
#         class_name, confidence, color = item

#         current_y = (
#             legend_y
#             + index * line_height
#         )

#         cv2.rectangle(
#             mask_canvas,
#             (legend_x, current_y - 18),
#             (legend_x + 28, current_y + 8),
#             color,
#             -1
#         )

#         cv2.putText(
#             mask_canvas,
#             f"{class_name} ({confidence:.2f})",
#             (legend_x + 40, current_y + 5),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.62,
#             (255, 255, 255),
#             2,
#             cv2.LINE_AA
#         )

#     return mask_canvas
def create_segmentation_mask_image(
    roi,
    result
):
    """
    Membuat gambar segmentation mask.

    Legend hanya menampilkan SATU BARIS untuk setiap class.
    Jika ada banyak object dengan class yang sama,
    maka hanya jumlah object dan confidence tertinggi yang ditampilkan.
    """

    height, width = roi.shape[:2]

    mask_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    mask_canvas[:] = (20, 80, 20)

    # Menyimpan data setiap class
    legend_items = {}

    # ==========================================================
    # SEGMENTATION MODEL
    # ==========================================================
    if (
        result.masks is not None
        and result.boxes is not None
    ):

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes

        object_count = min(len(masks), len(boxes))

        for index in range(object_count):

            class_index = int(boxes.cls[index].item())
            confidence = float(boxes.conf[index].item())

            if confidence < CONF_THRESHOLD:
                continue

            class_name = get_model_class_name(class_index)
            color = get_class_color(class_name, class_index)

            resized_mask = cv2.resize(
                masks[index],
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

            binary_mask = resized_mask > 0.5
            mask_canvas[binary_mask] = color

            # Simpan data legend
            if class_name not in legend_items:
                legend_items[class_name] = {
                    "count": 1,
                    "confidence": confidence,
                    "color": color
                }
            else:
                legend_items[class_name]["count"] += 1
                legend_items[class_name]["confidence"] = max(
                    legend_items[class_name]["confidence"],
                    confidence
                )

    # ==========================================================
    # DETECTION MODEL (fallback)
    # ==========================================================
    elif result.boxes is not None:

        for box in result.boxes:

            class_index = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            if confidence < CONF_THRESHOLD:
                continue

            class_name = get_model_class_name(class_index)
            color = get_class_color(class_name, class_index)

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            cv2.rectangle(
                mask_canvas,
                (x1, y1),
                (x2, y2),
                color,
                -1
            )

            if class_name not in legend_items:
                legend_items[class_name] = {
                    "count": 1,
                    "confidence": confidence,
                    "color": color
                }
            else:
                legend_items[class_name]["count"] += 1
                legend_items[class_name]["confidence"] = max(
                    legend_items[class_name]["confidence"],
                    confidence
                )

    # ==========================================================
    # DRAW LEGEND
    # ==========================================================

    legend_x = 20
    line_height = 36

    visible_items = list(legend_items.items())

    legend_height = len(visible_items) * line_height

    legend_y = max(
        30,
        height - legend_height - 20
    )

    for index, (class_name, info) in enumerate(visible_items):

        current_y = legend_y + index * line_height

        cv2.rectangle(
            mask_canvas,
            (legend_x, current_y - 18),
            (legend_x + 28, current_y + 8),
            info["color"],
            -1
        )

        cv2.putText(
            mask_canvas,
            f"{class_name} ({info['count']})  {info['confidence']:.2f}",
            (legend_x + 40, current_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return mask_canvas

def draw_summary_panel(
    image,
    total_sampah,
    organik_count,
    organik_percentage,
    anorganik_count,
    anorganik_percentage
):
    """
    Menambahkan informasi total, organik, dan anorganik
    pada panel hasil deteksi.
    """

    output = image.copy()
    overlay = output.copy()

    panel_x1 = 10
    panel_y1 = 10
    panel_x2 = min(
        output.shape[1] - 10,
        280
    )
    panel_y2 = min(
        output.shape[0] - 10,
        65
    )

    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.60,
        output,
        0.40,
        0,
        output
    )

    summary_lines = [
        (
            f"Total: {total_sampah}",
            (0, 255, 255),
            0.90
        )
    ]

    current_y = 48

    for text, color, font_scale in summary_lines:
        cv2.putText(
            output,
            text,
            (25, current_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA
        )

        current_y += 40

    return output


def create_detection_report_image(
    roi,
    result,
    total_sampah,
    organik_count,
    organik_percentage,
    anorganik_count,
    anorganik_percentage
):
    """
    Membuat satu gambar report yang menyatu:

    Kiri:
    - Hasil deteksi.
    - Bounding box.
    - Segmentation.
    - Label.
    - Total organik dan anorganik.

    Kanan:
    - Segmentation mask.
    """

    annotated_roi = result.plot()

    annotated_roi = draw_summary_panel(
        annotated_roi,
        total_sampah,
        organik_count,
        organik_percentage,
        anorganik_count,
        anorganik_percentage
    )

    segmentation_mask = (
        create_segmentation_mask_image(
            roi,
            result
        )
    )

    (
        annotated_roi,
        segmentation_mask
    ) = resize_to_same_height(
        annotated_roi,
        segmentation_mask
    )

    left_panel = add_title(
        annotated_roi,
        "Detection Label & Mask"
    )

    right_panel = add_title(
        segmentation_mask,
        ""
    )

    # Langsung digabung tanpa gap agar kedua bagian tersambung.
    final_report = np.hstack(
        (
            left_panel,
            right_panel
        )
    )

    return final_report


# =========================================================
# COUNT DETECTIONS
# =========================================================

def count_detected_waste(result):
    """
    Menghitung object organik dan anorganik berdasarkan
    class index model YOLO.
    """

    organik_count = 0
    anorganik_count = 0
    detected_items = []

    if result.boxes is None:
        return (
            0,
            0,
            0,
            0.0,
            0.0,
            []
        )

    for box in result.boxes:
        class_index = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        if confidence < CONF_THRESHOLD:
            continue

        class_name = get_model_class_name(
            class_index
        )

        if class_index in ORGANIK_CLASSES:
            organik_count += 1

            detected_items.append(
                {
                    "type": "organik",
                    "class_name": class_name,
                    "confidence": confidence
                }
            )

        elif class_index in ANORGANIK_CLASSES:
            anorganik_count += 1

            detected_items.append(
                {
                    "type": "anorganik",
                    "class_name": class_name,
                    "confidence": confidence
                }
            )

    total_sampah = (
        organik_count
        + anorganik_count
    )

    if total_sampah > 0:
        organik_percentage = (
            organik_count
            / total_sampah
            * 100
        )

        anorganik_percentage = (
            anorganik_count
            / total_sampah
            * 100
        )

    else:
        organik_percentage = 0.0
        anorganik_percentage = 0.0

    return (
        total_sampah,
        organik_count,
        anorganik_count,
        organik_percentage,
        anorganik_percentage,
        detected_items
    )


# =========================================================
# TELEGRAM CAPTION
# =========================================================

def create_telegram_caption(
    total_sampah,
    organik_count,
    organik_percentage,
    anorganik_count,
    anorganik_percentage
):
    """
    Membuat caption HTML untuk Telegram.
    """

    return (
        "🚨 <b>Sampah Terdeteksi</b>\n\n"

        f"📍 Lokasi: <b>{LOCATION_NAME}</b>\n"

        f'<a href="{MAP_URL}">'
        "🗺️ Buka lokasi di Google Maps"
        "</a> \n\n"

        f"🗑️ Total sampah: <b>{total_sampah}</b>\n"
        f"🍃 Organik: <b>{organik_count}</b> "
        f"({organik_percentage:.1f}%)\n"

        f"🧴 Anorganik: <b>{anorganik_count}</b> "
        f"({anorganik_percentage:.1f}%)\n\n"
        f"© RPI Sistem Monitoring Sampah"
    )


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():
    stream = RTSPStream(RTSP_URL)

    last_sent_time = 0
    cooldown_warning_sent = False

    print("✅ YOLO sampah monitoring started.")
    print("Tekan ESC untuk keluar.")
    print(
        f"⏱️ Telegram cooldown: "
        f"{COOLDOWN} detik / "
        f"{COOLDOWN // 60} menit"
    )

    try:
        while True:
            ret, frame = stream.read()

            if not ret or frame is None:
                if cv2.waitKey(1) & 0xFF == 27:
                    break

                time.sleep(0.01)
                continue

            (
                roi,
                roi_x1,
                roi_y1,
                roi_x2,
                roi_y2
            ) = get_roi(frame)

            if roi is None or roi.size == 0:
                print(
                    "⚠️ ROI kosong. "
                    "Periksa konfigurasi ROI."
                )

                time.sleep(0.1)
                continue

            clean_frame = frame.copy()
            clean_roi = roi.copy()

            # =================================================
            # YOLO INFERENCE
            # =================================================

            results = model(
                clean_roi,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMG_SIZE,
                verbose=False
            )

            result = results[0]

            (
                total_sampah,
                organik_count,
                anorganik_count,
                organik_percentage,
                anorganik_percentage,
                detected_items
            ) = count_detected_waste(result)

            # =================================================
            # CREATE REPORT
            # =================================================

            detection_report = (
                create_detection_report_image(
                    clean_roi,
                    result,
                    total_sampah,
                    organik_count,
                    organik_percentage,
                    anorganik_count,
                    anorganik_percentage
                )
            )

            # =================================================
            # TERMINAL STATUS
            # =================================================

            print(
                "\r"
                f"Total: {total_sampah} | "
                f"Organik: {organik_count} | "
                f"Anorganik: {anorganik_count}",
                end="",
                flush=True
            )

            # =================================================
            # TELEGRAM ALERT
            # =================================================

            current_time = time.time()

            cooldown_passed = (
                current_time
                - last_sent_time
            ) >= COOLDOWN

            if total_sampah >= TRIGGER_COUNT:
                if cooldown_passed:
                    print(
                        f"\n🚨 Sampah terdeteksi: "
                        f"{total_sampah}"
                    )

                    caption = create_telegram_caption(
                        total_sampah,
                        organik_count,
                        organik_percentage,
                        anorganik_count,
                        anorganik_percentage
                    )

                    sent = send_two_photos(
                        clean_frame,
                        detection_report,
                        caption
                    )

                    if sent:
                        last_sent_time = current_time
                        cooldown_warning_sent = False

                        print(
                            "✅ Cooldown dimulai. "
                            "Alert berikutnya dikirim "
                            "setelah cooldown selesai."
                        )

                else:
                    remaining_seconds = max(
                        0,
                        int(
                            COOLDOWN
                            - (
                                current_time
                                - last_sent_time
                            )
                        )
                    )

                    if (
                        remaining_seconds <= 600
                        and not cooldown_warning_sent
                    ):
                        cooldown_warning_sent = True

                        print(
                            "\n⏳ Sampah masih terdeteksi, "
                            "tetapi cooldown belum selesai. "
                            f"Sisa {remaining_seconds} detik."
                        )

            # Reset warning jika jumlah sampah kembali normal.
            if total_sampah < TRIGGER_COUNT:
                cooldown_warning_sent = False

            # =================================================
            # DISPLAY
            # =================================================

            if SHOW_PREVIEW:
                if SHOW_ROI_DEBUG:
                    debug_frame = frame.copy()

                    cv2.rectangle(
                        debug_frame,
                        (roi_x1, roi_y1),
                        (roi_x2, roi_y2),
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        debug_frame,
                        "Detection ROI",
                        (
                            roi_x1 + 10,
                            roi_y1 + 30
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    cv2.imshow(
                        "Original Frame + ROI",
                        debug_frame
                    )

                cv2.imshow(
                    "Detection Report",
                    detection_report
                )

                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        print(
            "\n🛑 Program dihentikan oleh pengguna."
        )

    except Exception as error:
        print(
            f"\n❌ Main program error: {error}"
        )

        raise

    finally:
        stream.release()
        cv2.destroyAllWindows()

        print(
            "\n🛑 Monitoring stopped."
        )


if __name__ == "__main__":
    main()