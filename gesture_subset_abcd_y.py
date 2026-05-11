"""Reduced gesture subset for this ASL glove project.

This file defines the smaller gesture set to use for training and inference.
Use it when you want to limit the model to only the selected gestures.
"""

REDUCED_GESTURE_SET = ["A", "B", "C", "D", "Y"]


def get_reduced_gesture_set() -> list[str]:
    """Return the reduced gesture set."""
    return list(REDUCED_GESTURE_SET)


def is_allowed_gesture(label: str) -> bool:
    """Return True if the label is part of the reduced gesture set."""
    return label in REDUCED_GESTURE_SET


def parse_label_list(labels: str | None) -> list[str] | None:
    """Parse a comma-separated label string into a list of labels."""
    if labels is None:
        return None
    label_list = [lab.strip() for lab in labels.split(",") if lab.strip()]
    return label_list or None
