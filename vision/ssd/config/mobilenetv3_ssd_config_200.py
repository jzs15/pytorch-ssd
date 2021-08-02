import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 200
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(13, 15, SSDBoxSizes(40, 68), [2, 3]),
    SSDSpec(7, 29, SSDBoxSizes(68, 96), [2, 3]),
    SSDSpec(4, 50, SSDBoxSizes(96, 124), [2, 3]),
    SSDSpec(2, 100, SSDBoxSizes(124, 152), [2, 3]),
    SSDSpec(1, 200, SSDBoxSizes(152, 180), [2, 3]),
    SSDSpec(1, 200, SSDBoxSizes(180, 200), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)