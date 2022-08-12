import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(19, 16, SSDBoxSizes(30, 60), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(105, 164), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(164, 216), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(216, 264), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(264, 300), [2, 3])
]


priors = generate_ssd_priors(specs, image_size)