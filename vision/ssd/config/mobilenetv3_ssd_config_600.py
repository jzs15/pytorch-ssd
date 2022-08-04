import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 600
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(38, 16, SSDBoxSizes(60, 120), [2, 3]),
    SSDSpec(19, 32, SSDBoxSizes(120, 220), [2, 3]),
    SSDSpec(10, 60, SSDBoxSizes(220, 320), [2, 3]),
    SSDSpec(5, 120, SSDBoxSizes(320, 420), [2, 3]),
    SSDSpec(3, 200, SSDBoxSizes(420, 520), [2, 3]),
    SSDSpec(2, 300, SSDBoxSizes(520, 620), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)