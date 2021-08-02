import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 160
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(10, 16, SSDBoxSizes(32, 54), [2, 3]),
    SSDSpec(5, 32, SSDBoxSizes(54, 77), [2, 3]),
    SSDSpec(3, 54, SSDBoxSizes(77, 99), [2, 3]),
    SSDSpec(2, 80, SSDBoxSizes(99, 122), [2, 3]),
    SSDSpec(1, 160, SSDBoxSizes(122, 144), [2, 3]),
    SSDSpec(1, 160, SSDBoxSizes(144, 160), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)