import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 400
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(25, 16, SSDBoxSizes(80, 140), [2, 3]),
    SSDSpec(13, 32, SSDBoxSizes(140, 200), [2, 3]),
    SSDSpec(7, 60, SSDBoxSizes(200, 260), [2, 3]),
    SSDSpec(4, 100, SSDBoxSizes(260, 320), [2, 3]),
    SSDSpec(2, 200, SSDBoxSizes(320, 380), [2, 3]),
    SSDSpec(1, 400, SSDBoxSizes(380, 440), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)