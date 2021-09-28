import tensorflow as tf
from vggnet import vgg_16, roi_fc
from ops import relu, conv, rpn2proposal, fully_connected
RATIO = [0.5, 1.0, 2.0]
SCALE = [128, 256, 512]

def rpn(inputs):
    num_anchors = len(SCALE) * len(RATIO)
    with tf.variable_scope("rpn"):
        inputs = relu(conv("conv1", inputs, 512, 3, 1))
        cls = conv("cls", inputs, num_anchors * 2, 1, 1)
        reg = conv("reg", inputs, num_anchors * 4, 1, 1)
    first_dim = tf.shape(cls)[0]
    cls = tf.reshape(cls, [first_dim, -1, 2])
    reg = tf.reshape(reg, [first_dim, -1, 4])
    return cls, reg

def unified_net(inputs, anchors, CLASSES,  NUMS_PROPOSAL, NMS_THRESHOLD, IMG_H, IMG_W):
    vgg_logits = vgg_16(inputs)
    print("inputs: ",vgg_logits)
    rpn_cls, rpn_reg = rpn(vgg_logits)
    print("rpn_cls: ",rpn_cls)
    normal_bbox, reverse_bbox, bbox_idx = rpn2proposal(rpn_cls, rpn_reg, anchors, NUMS_PROPOSAL, NMS_THRESHOLD, IMG_H, IMG_W)
    print("normal_bbox: ",normal_bbox)
    print("reverse_bbox: ",reverse_bbox)
    print("bbox_idx: ",bbox_idx)
    inputs = roi_fc(vgg_logits, reverse_bbox, bbox_idx)
    print("inputs: ",inputs)
    inputs = tf.squeeze(inputs, axis=[1, 2])
    print("inputs: ",inputs)
    cls = fully_connected("classification", inputs, len(CLASSES)+1)
    reg = fully_connected("regression", inputs, 4)
    return cls, reg, normal_bbox,vgg_logits