# 🎯 Real-time Object Counter

> **30 Days of AI Projects**

Detect and count objects in real-time using your **webcam or any video file** — powered by YOLOv8. Displays live bounding boxes, labels, confidence scores, and a per-class count overlay. Also includes a **Streamlit dashboard** for uploading and analyzing videos.

---

## 🎬 Demo

```
Webcam feed with live overlay:

┌─────────────────────────────────┬──────────────┐
│                                 │ LIVE COUNT   │
│   [person 94%]                  │ car:    2    │
│   ┌──────────┐                  │ person: 1    │
│   │          │  [car 88%]       │              │
│   └──────────┘  ┌─────────┐    │              │
│                 └─────────┘    │              │
└─────────────────────────────────┴──────────────┘
Press q=quit | s=screenshot | c=clear counts
```

---

## ✨ Features

- 📸 Works with **webcam** or any **video file** (mp4, avi, mov)
- 🏷️ Detects **80 object classes** (COCO dataset) out of the box
- 📊 Live count overlay panel in the corner
- 🎨 Color-coded boxes per object type
- 💾 Save annotated video with `--save`
- 📷 Screenshot with `s` key
- 🌐 Streamlit dashboard for video upload & analysis
- 🧪 Unit tests included

---

## 📁 Project Structure

```
day3-realtime-object-counter/
├── app/
│   ├── counter.py          # Main OpenCV + YOLOv8 script (webcam/video)
│   └── dashboard.py        # Streamlit web dashboard
├── config/
│   ├── __init__.py
│   └── settings.py         # Model name, confidence threshold, colors
├── tests/
│   └── test_counter.py     # Unit tests
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/day3-realtime-object-counter.git
cd day3-realtime-object-counter
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **First run** will auto-download `yolov8n.pt` (~6MB) from Ultralytics. No API key needed!

---

## ▶️ Run — Webcam (Live)

```bash
python app/counter.py --source 0
```

---

## 🎬 Run — Video File

```bash
python app/counter.py --source path/to/video.mp4
```

### Save the annotated output:

```bash
python app/counter.py --source path/to/video.mp4 --save --output result.mp4
```

---

## ⌨️ Keyboard Controls

| Key | Action                              |
| --- | ----------------------------------- |
| `q` | Quit                                |
| `s` | Save screenshot as `screenshot.png` |
| `c` | Clear all counts                    |

---

## 🌐 Run the Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

Opens at `http://localhost:8501`

- Upload any video file
- Filter by specific object classes
- View per-frame counts and session summary metrics

---

## ⚙️ Configuration

Edit `config/settings.py` to customize:

```python
MODEL_NAME = "yolov8n.pt"       # fastest; swap to yolov8m.pt for more accuracy
CONFIDENCE_THRESHOLD = 0.5      # raise to reduce false positives
```

### Model Options (trade speed vs accuracy)

| Model        | Size  | Speed          | Accuracy |
| ------------ | ----- | -------------- | -------- |
| `yolov8n.pt` | 6 MB  | ⚡⚡⚡ Fastest | ★★☆      |
| `yolov8s.pt` | 22 MB | ⚡⚡ Fast      | ★★★      |
| `yolov8m.pt` | 52 MB | ⚡ Medium      | ★★★★     |
| `yolov8l.pt` | 87 MB | 🐢 Slow        | ★★★★★    |

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

---

## 🐛 Common Errors & Fixes

| Error                              | Fix                                                    |
| ---------------------------------- | ------------------------------------------------------ |
| `Cannot open source: 0`            | No webcam found — try `--source 1` or use a video file |
| `ModuleNotFoundError: ultralytics` | Run `pip install -r requirements.txt`                  |
| Slow FPS                           | Switch to `yolov8n.pt` in `config/settings.py`         |
| Black screen on Mac                | Allow camera permissions in System Settings → Privacy  |

---

## 💡 What I Learned

- How YOLO (You Only Look Once) object detection works
- How to read video frames with OpenCV `VideoCapture`
- How to draw bounding boxes and text overlays with OpenCV
- How to use `argparse` for a proper CLI interface
- How to build a Streamlit dashboard for video processing

---


## 📄 License

MIT License — free to use, modify, and build on!
