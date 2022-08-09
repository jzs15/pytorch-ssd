import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 480
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(30, 16, SSDBoxSizes(96, 168), [2, 3]),
    SSDSpec(15, 32, SSDBoxSizes(168, 240), [2, 3]),
    SSDSpec(8, 60, SSDBoxSizes(214, 312), [2, 3]),
    SSDSpec(4, 120, SSDBoxSizes(312, 384), [2, 3]),
    SSDSpec(2, 240, SSDBoxSizes(384, 456), [2, 3]),
    SSDSpec(1, 280, SSDBoxSizes(456, 528), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)