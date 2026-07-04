"""
Streamlit dashboard for Real-time Object Counter.
Upload a video or use webcam to count objects live.
"""

import cv2
import tempfile
import time
import streamlit as st
from collections import Counter
from ultralytics import YOLO
from config.settings import CONFIDENCE_THRESHOLD, MODEL_NAME, COLORS

st.set_page_config(
    page_title="Object Counter",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Real-time Object Counter")
st.caption("Powered by YOLOv8 · Day 3 of 30 Days of AI Projects")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Confidence threshold", 0.1, 1.0,
                           CONFIDENCE_THRESHOLD, 0.05)
    target_objects = st.multiselect(
        "Filter objects (leave empty = show all)",
        options=["person", "car", "dog", "cat", "bicycle", "motorcycle",
                 "bus", "truck", "chair", "bottle", "cell phone", "laptop"],
        default=[]
    )
    st.divider()
    st.info("**Keys (in OpenCV window):**\n- `q` quit\n- `s` screenshot\n- `c` clear counts")


@st.cache_resource
def load_model():
    return YOLO(MODEL_NAME)


source_tab, upload_tab = st.tabs(["📹 Webcam", "📁 Upload Video"])

with upload_tab:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        video_path = tfile.name

        model = load_model()
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        col1, col2 = st.columns([2, 1])
        frame_display = col1.empty()
        count_display = col2.empty()
        progress = st.progress(0)

        all_counts = Counter()
        frame_num = 0

        run_btn = st.button("▶️ Run Detection", type="primary")
        if run_btn:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, verbose=False)
                frame_counts = Counter()

                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        label = result.names[cls_id]
                        conf = float(box.conf[0])

                        if conf < confidence:
                            continue
                        if target_objects and label not in target_objects:
                            continue

                        frame_counts[label] += 1
                        all_counts[label] += 1

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = COLORS.get(label, (0, 255, 0))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} {conf:.0%}",
                                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (255, 255, 255), 1)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_display.image(frame_rgb, use_column_width=True)

                count_display.markdown("### 📊 Frame Counts")
                for name, cnt in sorted(frame_counts.items()):
                    count_display.metric(name, cnt)

                frame_num += 1
                progress.progress(min(frame_num / total_frames, 1.0))

            cap.release()
            st.success("✅ Processing complete!")

            st.markdown("### 🏁 Session Summary")
            cols = st.columns(min(len(all_counts), 4))
            for i, (name, cnt) in enumerate(
                    sorted(all_counts.items(), key=lambda x: -x[1])):
                cols[i % len(cols)].metric(name.capitalize(), cnt)

with source_tab:
    st.info("👆 Run the OpenCV window version for webcam:\n```\npython app/counter.py --source 0\n```")
    st.markdown("The webcam feed requires running the app locally via the terminal command above.")
