import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 480
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(30, 16, SSDBoxSizes(48, 95), [2, 3]),
    SSDSpec(15, 32, SSDBoxSizes(95, 176), [2, 3]),
    SSDSpec(8, 60, SSDBoxSizes(176, 256), [2, 3]),
    SSDSpec(4, 120, SSDBoxSizes(256, 336), [2, 3]),
    SSDSpec(2, 240, SSDBoxSizes(336, 416), [2, 3]),
    SSDSpec(1, 280, SSDBoxSizes(416, 496), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)