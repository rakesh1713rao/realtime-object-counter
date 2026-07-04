"""
Tests for object counter utility functions.
"""

import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import CONFIDENCE_THRESHOLD, COLORS


def test_confidence_threshold_in_range():
    assert 0.0 < CONFIDENCE_THRESHOLD < 1.0, \
        "CONFIDENCE_THRESHOLD must be between 0 and 1"


def test_colors_are_bgr_tuples():
    for label, color in COLORS.items():
        assert len(color) == 3, f"Color for '{label}' must be a 3-tuple (BGR)"
        assert all(0 <= c <= 255 for c in color), \
            f"Color values for '{label}' must be 0–255"


def test_draw_boxes_does_not_crash():
    """draw_boxes should handle an empty results list without error."""
    import cv2
    from app.counter import draw_boxes

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_boxes(frame, [], {})  # empty results → no crash
    assert frame.shape == (480, 640, 3)


def test_colors_dict_has_common_classes():
    expected = {"person", "car", "dog", "cat"}
    assert expected.issubset(set(COLORS.keys())), \
        "COLORS dict should contain at least person, car, dog, cat"
