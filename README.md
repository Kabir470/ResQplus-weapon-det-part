# YOLO11 Gun Detection

Simple and fast gun/weapon detection using Ultralytics YOLO (model.pt). Includes:

- gun-detection-only.py — minimal, webcam/stream viewer for gun detection
- main.py — ESP32-CAM stream + on-detect alert endpoint + performance tweaks
- train.py — quick-start to fine-tune YOLO on your dataset

## Requirements

- Python 3.12 (verified)
- Packages:
  - ultralytics
  - opencv-python
  - cvzone
  - numpy

Install dependencies:

```powershell
# From the project root
pip install -r requirements.txt
```

Note on GPU: Your system uses an AMD GPU. Ultralytics typically accelerates with CUDA (NVIDIA). On Windows with AMD, PyTorch CUDA is unavailable. CPU works out of the box; advanced AMD acceleration via DirectML may not be fully supported by Ultralytics. If you try DirectML, do so experimentally.

## Files

- `main.py` — ESP32-CAM streaming + detection + alert HTTP request when a weapon is detected. Resizes the display window to 640x640 and uses threaded detection every few frames.
- `gun-detection-only.py` — clean, focused script; opens a video source (ESP32 or webcam), draws red boxes for detected guns, prints confidence.
- `train.py` — trains a YOLO11 model on your dataset (edit the `data.yaml` path).
- `model.pt` — your trained YOLO model weights (place in the project root).
- `requirements.txt` — Python dependencies.

## Quick start

### 1) Minimal gun detection (current webcam)

```powershell
# Uses your default webcam (index 0)
python .\gun-detection-only.py
```

- Press `q` or `Esc` to exit.
- To use a video file instead of a webcam, edit the script and set:
  ```python
  cap = cv2.VideoCapture("path-to-video.mp4")
  ```

### 2) Minimal gun detection from ESP32-CAM

`gun-detection-only.py` is already configured to read from your ESP32:

- Stream URL: `http://192.168.0.107:81/stream`
- The script will auto-reconnect if the stream drops.

Run:

```powershell
python .\gun-detection-only.py
```

If you prefer to keep your webcam for the minimal script, switch to `main.py` for ESP32 (see below).

### 3) ESP32-CAM detection + alerts (main.py)

`main.py` connects to the ESP32-CAM stream and can notify an endpoint when a weapon is detected.

Edit these lines to match your device:

```python
ESP32_IMG_URL = 'http://<your-esp32-ip>:81/stream'
ESP32_ALERT_URL = 'http://<your-esp32-ip>/alert'
```

Run:

```powershell
python .\main.py
```

Notes:

- The window displays at 640x640.
- The script skips frames and runs inference in a background thread for smoother display.
- Console logs show all classes detected and will highlight weapon detections.

### 4) Training

Edit `train.py` to point to your dataset YAML:

```python
model = YOLO("yolo11n.pt")
data_path = r"E:/downloads/gun img/theos-guns/theos-guns/data.yaml"
```

Then start training:

```powershell
python .\train.py
```

This will save new weights (e.g., `runs/detect/train*/weights/best.pt`). Rename or copy the trained weights to `model.pt` in the project root to use them for detection.

## Troubleshooting

- "Could not open ESP32-CAM stream"

  - Ensure the ESP32-CAM is powered and connected to the same network
  - Verify the IP address and that `http://<ip>:81/stream` is reachable in a browser
  - Some routers block cross-subnet streams — use the same subnet as your PC

- "Everything is detected as a gun"

  - We restrict "weapon" marking to actual weapon-labeled classes only
  - Increase confidence threshold in scripts if needed (e.g., to 0.7)
  - Verify your `model.pt` is trained on the expected class list (e.g., `['Gun']`)

- AMD GPU not used
  - On Windows, CUDA acceleration is NVIDIA-only.
  - CPU works by default; for AMD acceleration you may explore PyTorch DirectML, but Ultralytics may not fully support it. Expect CPU as the reliable path.

## Tips

- Change the video source by updating `cv2.VideoCapture(...)` in the scripts.
- Window is set to 640x640 for readability; adjust as you like.
- To view class names known by the model: `print(model.names)`.

---

If you want, I can add screenshots/gifs or a small CLI to switch between webcam and ESP32 sources automatically.
