import numpy as np
import pytest
from detect_wetness import check_lights_on

def test_check_lights_on_all_dark():
    img = np.zeros((100, 100), dtype=np.uint8)
    assert not check_lights_on(img)

def test_check_lights_on_all_bright():
    img = np.full((100, 100), 255, dtype=np.uint8)
    assert check_lights_on(img)

def test_check_lights_on_half_bright():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:50, :] = 255
    assert check_lights_on(img)

def test_check_lights_on_below_threshold():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:40, :] = 255  # 40% bright
    assert not check_lights_on(img)

def test_check_lights_on_custom_threshold():
    img = np.full((100, 100), 180, dtype=np.uint8)
    # Should not trigger with default threshold, but will with lower brightness_threshold
    assert not check_lights_on(img)
    assert check_lights_on(img, brightness_threshold=150)
