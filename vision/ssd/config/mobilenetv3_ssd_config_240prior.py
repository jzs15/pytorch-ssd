import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 240
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(15, 16, SSDBoxSizes(48, 82), [pow(1060/600, 2), pow(1060/475, 2), pow(1060/320, 2), pow(595/430, 2)]),
    SSDSpec(8, 30, SSDBoxSizes(82, 115), [pow(1060/600, 2), pow(1060/475, 2), pow(1060/320, 2), pow(595/430, 2)]),
    SSDSpec(4, 60, SSDBoxSizes(115, 149), [pow(1060/600, 2), pow(1060/475, 2), pow(1060/320, 2), pow(595/430, 2)]),
    SSDSpec(2, 120, SSDBoxSizes(149, 182), [pow(1060/600, 2), pow(1060/475, 2), pow(1060/320, 2), pow(595/430, 2)]),
    SSDSpec(1, 240, SSDBoxSizes(182, 216), [pow(1060/600, 2), pow(1060/475, 2), pow(1060/320, 2), pow(595/430, 2)]),
    SSDSpec(1, 240, SSDBoxSizes(216, 240), [pow(1060/600, 2), pow(1060/475, 2), pow(1060/320, 2), pow(595/430, 2)])
]

priors = generate_ssd_priors(specs, image_size)