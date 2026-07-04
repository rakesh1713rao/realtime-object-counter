"""
Real-time Object Counter
Uses YOLOv8 to detect and count objects from webcam or video file.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import argparse
from collections import Counter
from ultralytics import YOLO
from config.settings import CONFIDENCE_THRESHOLD, MODEL_NAME, COLORS


def load_model(model_name: str = MODEL_NAME) -> YOLO:
    """Load YOLOv8 model. Downloads automatically on first run."""
    print(f"[INFO] Loading model: {model_name}")
    return YOLO(model_name)


def draw_boxes(frame, results, counts: dict) -> None:
    """Draw bounding boxes, labels, and live count overlay on frame."""
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            conf = float(box.conf[0])

            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS.get(label, (0, 255, 0))

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label pill
            text = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Count overlay panel (top-right)
    panel_x = frame.shape[1] - 220
    cv2.rectangle(frame, (panel_x - 10, 10), (frame.shape[1] - 10, 30 + len(counts) * 28),
                  (20, 20, 20), -1)
    cv2.putText(frame, "LIVE COUNT", (panel_x, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    for i, (name, count) in enumerate(sorted(counts.items())):
        color = COLORS.get(name, (0, 255, 0))
        cv2.putText(frame, f"{name}: {count}", (panel_x, 56 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def run(source=0, save_output: bool = False, output_path: str = "output.mp4"):
    """
    Run the real-time object counter.

    Args:
        source: 0 for webcam, or path to a video file (e.g. 'video.mp4')
        save_output: Whether to save the annotated video
        output_path: Path to save the output video
    """
    model = load_model()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    writer = None
    if save_output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        print(f"[INFO] Saving output to: {output_path}")

    print("[INFO] Press 'q' to quit | 'c' to clear counts | 's' to screenshot")

    all_time_counts = Counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        # Count objects in this frame
        frame_counts = Counter()
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                if float(box.conf[0]) >= CONFIDENCE_THRESHOLD:
                    frame_counts[label] += 1
                    all_time_counts[label] += 1

        draw_boxes(frame, results, frame_counts)

        # FPS overlay
        cv2.putText(frame, "Day 3 - Object Counter", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Real-time Object Counter", frame)

        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            all_time_counts.clear()
            print("[INFO] Counts cleared.")
        elif key == ord("s"):
            cv2.imwrite("screenshot.png", frame)
            print("[INFO] Screenshot saved: screenshot.png")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n===== Session Summary =====")
    for name, count in sorted(all_time_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:20s} {count:>5} detections")
    print("===========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Object Counter using YOLOv8")
    parser.add_argument("--source", default=0,
                        help="Video source: 0 for webcam, or path to video file")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output video")
    parser.add_argument("--output", default="output.mp4",
                        help="Output video filename (default: output.mp4)")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    run(source=source, save_output=args.save, output_path=args.output)
