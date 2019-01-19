import numpy as np


def calculate_pixel_change(curr, prev):
    abs_diff = int(np.sum(np.abs(curr - prev)))
    total_pixels = int(np.sum(np.ones(shape=curr.shape) * 255))
    print(curr.shape)
    return abs_diff / total_pixels
