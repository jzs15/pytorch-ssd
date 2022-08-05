import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 540
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(34, 16, SSDBoxSizes(54, 107), [2, 3]),
    SSDSpec(17, 32, SSDBoxSizes(107, 198), [2, 3]),
    SSDSpec(9, 60, SSDBoxSizes(198, 288), [2, 3]),
    SSDSpec(5, 108, SSDBoxSizes(288, 378), [2, 3]),
    SSDSpec(3, 180, SSDBoxSizes(378, 468), [2, 3]),
    SSDSpec(2, 270, SSDBoxSizes(468, 558), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)