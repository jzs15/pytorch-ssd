import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 360
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(23, 16, SSDBoxSizes(36, 72), [2, 3]),
    SSDSpec(12, 30, SSDBoxSizes(72, 126), [2, 3]),
    SSDSpec(6, 60, SSDBoxSizes(126, 195), [2, 3]),
    SSDSpec(3, 120, SSDBoxSizes(195, 260), [2, 3]),
    SSDSpec(2, 180, SSDBoxSizes(260, 318), [2, 3]),
    SSDSpec(1, 360, SSDBoxSizes(318, 360), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)