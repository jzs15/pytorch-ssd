import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 240
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(15, 16, SSDBoxSizes(24, 48), [2, 3]),
    SSDSpec(8, 30, SSDBoxSizes(48, 84), [2, 3]),
    SSDSpec(4, 60, SSDBoxSizes(84, 130), [2, 3]),
    SSDSpec(2, 120, SSDBoxSizes(130, 174), [2, 3]),
    SSDSpec(1, 240, SSDBoxSizes(174, 212), [2, 3]),
    SSDSpec(1, 240, SSDBoxSizes(212, 240), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)