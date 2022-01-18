import sys
import os
from ops import *
RATIO = [0.5, 1.0, 2.0]
SCALE = [128, 256, 512]

def rpn(inputs):
    num_anchors = len(SCALE) * len(RATIO)
    import pdb
    pdb.set_trace()
    with tf.variable_scope("rpn"):
        inputs = relu(conv("conv1", inputs, 512, 3, 1))
        cls = conv("cls", inputs, num_anchors * 2, 1, 1)
        reg = conv("reg", inputs, num_anchors * 4, 1, 1)
    first_dim = tf.shape(cls)[0]
    cls = tf.reshape(cls, [first_dim, -1, 2])
    reg = tf.reshape(reg, [first_dim, -1, 4])
    return cls, reg


