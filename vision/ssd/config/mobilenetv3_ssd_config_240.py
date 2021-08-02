import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 240
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(15, 16, SSDBoxSizes(48, 82), [2, 3]),
    SSDSpec(8, 30, SSDBoxSizes(82, 115), [2, 3]),
    SSDSpec(4, 60, SSDBoxSizes(115, 149), [2, 3]),
    SSDSpec(2, 120, SSDBoxSizes(149, 182), [2, 3]),
    SSDSpec(1, 240, SSDBoxSizes(182, 216), [2, 3]),
    SSDSpec(1, 240, SSDBoxSizes(216, 240), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)