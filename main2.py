import cv2
import numpy as np
import requests
import time
import threading
import math
from ultralytics import YOLO

# ================= CONFIG =================
# ESP32 stream and alert settings
ESP32_STREAM_URL = 'http://192.168.0.103:81/stream'   # your MJPEG stream
ESP32_IP = '192.168.0.103'                            # ESP32 IP (used for alert)
ESP32_ALERT_PORT = 8080                               # the alert server port on ESP32
ALERT_TOKEN = 'resqplus_secret'                      # must match ALERT_TOKEN on ESP32
ESP32_ALERT_PATH = '/alert'                           # endpoint on ESP32

# Build full alert URL (we'll use GET with token param)
ESP32_ALERT_URL = f'http://{ESP32_IP}:{ESP32_ALERT_PORT}{ESP32_ALERT_PATH}'

# YOLO configuration
MODEL_PATH = 'model.pt'            # Your trained YOLO11 model file
CONFIDENCE_THRESHOLD = 0.7        # model confidence threshold
NMS_THRESHOLD = 0.4               # IoU threshold
SKIP_FRAMES = 3                   # process every Nth frame
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 640

# Alert cooldown (seconds) to prevent repeated alerts
ALERT_COOLDOWN_SECONDS = 30       # change as needed

# ================== Globals ==================
frame_count = 0
latest_detections = []
detection_lock = threading.Lock()
last_alert_time_lock = threading.Lock()
last_alert_time = 0.0

# Load YOLO model
print("Loading YOLO model...")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("📝 Make sure 'model.pt' exists in the current directory")
    exit()

# class names from your model
classnames = ['Gun']  # adjust if your model uses different names

def send_alert_to_esp32():
    """Send HTTP GET to ESP32 alert endpoint with token (respects cooldown)."""
    global last_alert_time
    now = time.time()
    with last_alert_time_lock:
        if now - last_alert_time < ALERT_COOLDOWN_SECONDS:
            print(f"⏱️ Alert cooldown active ({int(ALERT_COOLDOWN_SECONDS - (now - last_alert_time))}s left). Skipping alert.")
            return
        # mark the last_alert_time immediately to avoid race conditions
        last_alert_time = now

    try:
        resp = requests.get(ESP32_ALERT_URL, params={'token': ALERT_TOKEN}, timeout=3)
        print(f"📡 Alert request sent -> {ESP32_ALERT_URL}?token=***  | Response: {resp.status_code} {resp.text}")
    except Exception as e:
        print("🚫 Failed to send alert to ESP32:", e)

def process_detection(frame):
    """Run YOLO inference and update detections. If weapon found, call ESP32 alert."""
    global latest_detections

    # Run model (Ultralytics API)
    try:
        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=NMS_THRESHOLD)
    except Exception as e:
        print("❌ Inference error:", e)
        return

    detections = []
    weapon_detected = False

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            # box.xyxy, box.conf, box.cls are tensors; convert to python types
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
            class_id = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)

            # Resolve class name
            class_name = classnames[class_id] if class_id < len(classnames) else f"class_{class_id}"

            # Consider weapon-related names
            weapon_keywords = ('gun','weapon','pistol','rifle','knife')
            is_weapon_class = any(k in class_name.lower() for k in weapon_keywords)

            # Debug print
            print(f"🔍 Detected: {class_name} (id={class_id}) conf={confidence:.2f} bbox=({x1},{y1},{x2},{y2})")

            if is_weapon_class and confidence >= CONFIDENCE_THRESHOLD:
                weapon_detected = True
                print(f"⚠️ WEAPON CLASSIFIED: {class_name} (conf {confidence:.2f})")

            detections.append({
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'coords': (x1, y1, x2, y2),
                'class_name': class_name,
                'confidence': confidence,
                'is_weapon': is_weapon_class and confidence >= CONFIDENCE_THRESHOLD
            })

    # Update shared detections
    with detection_lock:
        latest_detections = detections.copy()

    # If any weapon flagged, send alert (with cooldown)
    if weapon_detected:
        print("⚠️ Weapon detected — preparing to notify ESP32")
        send_alert_to_esp32()

# Open video stream
print("Attempting to connect to ESP32-CAM...")
cap = cv2.VideoCapture(ESP32_STREAM_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Error: Could not open ESP32-CAM stream")
    print("📝 Make sure the ESP32-CAM is on, IP is correct, stream URL is accessible")
    exit()

print("🔁 Starting video stream and YOLO detection...")

ret = True
detection_thread = None

try:
    while ret:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame from stream; reconnecting...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(ESP32_STREAM_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        frame_count += 1

        # Resize for display only
        display_frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

        # Run detection on every SKIP_FRAMES
        if frame_count % SKIP_FRAMES == 0:
            if detection_thread is None or not detection_thread.is_alive():
                # pass a copy so OpenCV buffer isn't reused
                detection_thread = threading.Thread(target=process_detection, args=(frame.copy(),))
                detection_thread.daemon = True
                detection_thread.start()

        # Draw latest detections
        with detection_lock:
            for detection in latest_detections:
                x1, y1, x2, y2 = detection['coords']
                class_name = detection['class_name']
                conf = detection['confidence']
                is_weapon = detection['is_weapon']

                # scale coordinates from original frame size to display size
                h_orig, w_orig = frame.shape[:2]
                scale_x = RESIZE_WIDTH / float(w_orig)
                scale_y = RESIZE_HEIGHT / float(h_orig)
                x1s = int(x1 * scale_x)
                y1s = int(y1 * scale_y)
                x2s = int(x2 * scale_x)
                y2s = int(y2 * scale_y)

                color = (0, 0, 255) if is_weapon else (0, 255, 0)
                cv2.rectangle(display_frame, (x1s, y1s), (x2s, y2s), color, 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(display_frame, label, (x1s, y1s - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show FPS
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow('ESP32-CAM YOLO Weapon Detection', display_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Exiting.")
