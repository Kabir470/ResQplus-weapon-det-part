import cv2
import numpy as np
import requests
import time
import threading
import math
from ultralytics import YOLO

# ESP32 URLs (change to your device's IP)
ESP32_IMG_URL = 'http://192.168.0.103:81/stream'
ESP32_ALERT_URL = 'http://192.168.0.103/alert'

# YOLO11 configuration
MODEL_PATH = 'model.pt'  # Your trained YOLO11 model
CONFIDENCE_THRESHOLD = 0.7  # Increased from 0.5 to be more selective
NMS_THRESHOLD = 0.4

# Performance optimization settings
SKIP_FRAMES = 3  # Process every 3rd frame for detection
RESIZE_WIDTH = 640  # Resize frame for faster processing
RESIZE_HEIGHT = 640  # Resize frame height
frame_count = 0

# Load YOLO11 model
print("Loading YOLO11 model...")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("📝 Make sure 'model.pt' exists in the current directory")
    exit()

# Custom weapon detection class names
classnames = ['Gun']  # Your model's class names

# Global variables for threading
latest_detections = []
detection_lock = threading.Lock()

def process_detection(frame):
    """Process YOLO11 detection in a separate thread"""
    global latest_detections
    
    # Run YOLO11 inference
    results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=NMS_THRESHOLD)
    
    detections = []
    weapon_detected = False
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence
                confidence = float(box.conf[0])
                
                # Get class
                class_id = int(box.cls[0])
                if class_id < len(classnames):
                    class_name = classnames[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                # Check if it's a weapon detection
                weapon_classes = ['gun', 'weapon', 'pistol', 'rifle', 'knife']
                is_weapon_class = any(weapon in class_name.lower() for weapon in weapon_classes)
                
                # Debug: Print all detections to understand what the model is detecting
                print(f"🔍 Model detected: {class_name} (Class ID: {class_id}) with confidence {confidence:.2f}")
                
                # Only treat as weapon if it's actually a weapon class
                if is_weapon_class:
                    weapon_detected = True
                    print(f"⚠️ WEAPON ALERT: {class_name} with confidence {confidence:.2f}")
                
                detections.append({
                    'bbox': (x1, y1, x2 - x1, y2 - y1),  # Convert to x, y, w, h format
                    'class_name': class_name,
                    'confidence': confidence,
                    'is_weapon': is_weapon_class,  # Only weapon classes are marked as weapons
                    'coords': (x1, y1, x2, y2)  # Keep original format for drawing
                })
    
    # Update global detections
    with detection_lock:
        latest_detections = detections.copy()
    
    # Send alert if weapon detected
    if weapon_detected:
        print("⚠️ Weapon detected!")
        try:
            resp = requests.get(ESP32_ALERT_URL, timeout=1)
            print("Alert sent! Response:", resp.status_code, resp.text)
        except Exception as e:
            print("🚫 Failed to notify ESP32:", e)

# Open video stream
print("Attempting to connect to ESP32-CAM...")
cap = cv2.VideoCapture(ESP32_IMG_URL)

# Set buffer size to reduce latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Check if camera opened successfully
if not cap.isOpened():
    print("❌ Error: Could not open ESP32-CAM stream")
    print("📝 Make sure:")
    print("   1. ESP32-CAM is powered on and connected to WiFi")
    print("   2. The IP address is correct")
    print("   3. The stream URL is accessible")
    exit()

print("🔁 Starting optimized video stream and YOLO11 detection...")

ret = True
detection_thread = None

while ret:
    start_time = time.time()
    ret, frame = cap.read()
    
    if ret:
        frame_count += 1
        
        # Resize frame for display to 640x640
        display_frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        
        # Only process detection every few frames
        if frame_count % SKIP_FRAMES == 0:
            # Start detection in background thread if not already running
            if detection_thread is None or not detection_thread.is_alive():
                detection_thread = threading.Thread(target=process_detection, args=(frame.copy(),))
                detection_thread.daemon = True
                detection_thread.start()
        
        # Draw latest detections on current frame
        with detection_lock:
            for detection in latest_detections:
                x1, y1, x2, y2 = detection['coords']
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # Scale coordinates to match resized display frame
                frame_height, frame_width = frame.shape[:2]
                scale_x = RESIZE_WIDTH / frame_width
                scale_y = RESIZE_HEIGHT / frame_height
                
                # Apply scaling to coordinates
                x1_scaled = int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)
                
                # Draw bounding box
                color = (0, 0, 255) if detection['is_weapon'] else (0, 255, 0)
                cv2.rectangle(display_frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
                
                # Add label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(display_frame, label, (x1_scaled, y1_scaled - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = 1.0 / elapsed_time
        else:
            fps = 0.0
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('ESP32-CAM YOLO11 Weapon Detection', display_frame)
        
        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        print("⚠️ Failed to read frame from stream")
        break

# Clean up
cap.release()
cv2.destroyAllWindows()