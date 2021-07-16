"""Default experiment config
"""
import cv2
from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# Define input shape of experiment
_C.INPUT_SHAPE = [608, 608, 3]

# Maximum number of detections to retain.
_C.MAX_OBJECTS = 500

# Minimum class probability, below which detections are pruned.
_C.CONF_THRESH = 0.18

# List of all categories that the model can predict.
_C.CATEGORIES = []

# Image resize rate.
_C.RESIZE_RATE = 1.0

# Altitude the photo was taken in meters.
_C.PHOTO_ALTITUDE = 120

# Resolution references in cm/px for different altitudes.
_C.RESOLUTION = []


def export_config(path):
    """Export config to a yaml file."""
    with open(path, "w") as f:
        f.write(_C.dump())


def load_config(path):
    """Load config yaml file."""
    with open(path, "r") as f:
        cfg = CN.load_cfg(f)
    return cfg
