import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 480
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(30, 16, SSDBoxSizes(48, 96), [2, 3]),
    SSDSpec(15, 32, SSDBoxSizes(96, 168), [2, 3]),
    SSDSpec(8, 60, SSDBoxSizes(168, 260), [2, 3]),
    SSDSpec(4, 120, SSDBoxSizes(260, 346), [2, 3]),
    SSDSpec(2, 240, SSDBoxSizes(346, 422), [2, 3]),
    SSDSpec(1, 480,  SSDBoxSizes(422, 480), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)