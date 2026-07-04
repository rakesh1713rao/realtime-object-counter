"""
Configuration settings for the Real-time Object Counter.
Tweak these values to adjust detection behaviour.
"""

# YOLOv8 model to use
# Options: yolov8n.pt (fastest), yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt (most accurate)
MODEL_NAME = "yolov8n.pt"

# Minimum confidence score to count a detection (0.0 – 1.0)
CONFIDENCE_THRESHOLD = 0.5

# BGR colors per object class (used for bounding boxes and labels)
COLORS = {
    "person":       (255, 100, 50),
    "car":          (50, 200, 255),
    "truck":        (50, 150, 255),
    "bus":          (50, 100, 255),
    "bicycle":      (100, 255, 100),
    "motorcycle":   (150, 255, 50),
    "dog":          (255, 200, 50),
    "cat":          (255, 150, 100),
    "chair":        (200, 100, 255),
    "bottle":       (100, 255, 200),
    "laptop":       (255, 50, 150),
    "cell phone":   (50, 255, 150),
}
