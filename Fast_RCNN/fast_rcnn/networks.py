import sys
import os
from vggnet import vgg_16
from ops import fully_connected


def network(inputs, boxes, box_idx, CLASSES):
    inputs = vgg_16(inputs, boxes, box_idx)
    cls = fully_connected("classification", inputs, len(CLASSES) + 1)
    reg = fully_connected("regression", inputs, 4)
    return cls, reg

