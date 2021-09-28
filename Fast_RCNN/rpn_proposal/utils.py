import sys
import os
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import os
import scipy.misc as scm

#from config import CLASSES, IMG_PATH, XML_PATH, MINIBATCH, BATCHSIZE, IMG_H, IMG_W
RATIO = [0.5, 1.0, 2.0]
SCALE = [128, 256, 512]

def crop_center(img, bboxes, cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    new_img = img[starty:starty+cropy,startx:startx+cropx]
    bboxes[:, 0]-=starty
    bboxes[:, 1]-=startx
    bboxes[:, 2]-=0
    bboxes[:, 3]-=0
    return new_img, bboxes

def read_data(xml_path, img_path, CLASSES, one_hot=True):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall("object")
    names = []
    gtbboxes = np.zeros([len(objects), 4], dtype=np.int32)
    for idx, obj in enumerate(objects):
        curr_name = obj.find("name").text
        if curr_name not in CLASSES:
            continue
        names.append(curr_name)
        xmin = int(obj.find("bndbox").find("xmin").text)
        xmax = int(obj.find("bndbox").find("xmax").text)
        ymin = int(obj.find("bndbox").find("ymin").text)
        ymax = int(obj.find("bndbox").find("ymax").text)
        gtbboxes[idx, 0] = (xmin + xmax)//2
        gtbboxes[idx, 1] = (ymin + ymax)//2
        gtbboxes[idx, 2] = xmax - xmin
        gtbboxes[idx, 3] = ymax - ymin
    #img = np.array(Image.open(img_path)) #
    img = np.array(scm.imread(img_path))
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = np.mean(img, axis=2)
    if one_hot:
        labels = np.zeros([len(objects), len(CLASSES)])
        for idx, name in enumerate(names):
            labels[idx, CLASSES.index(name)] = 1
    else:
        labels = np.zeros([len(objects)])
        for idx, name in enumerate(names):
            labels[idx] = CLASSES.index(name)
    return img, gtbboxes, labels

def generate_anchors(IMG_H, IMG_W):
    #anchors: [x, y, w, h]
    # IMG_H, IMG_W = img.shape[0], img.shape[1]
    h = int(np.ceil(IMG_H/16))
    w = int(np.ceil(IMG_W/16))
    anchor_wh = np.zeros([9, 2])
    count = 0
    for ratio in RATIO:
        for scale in SCALE:
            #ratio = h / w ---> h = ratio * w, area = h * w ---> area = ratio * w ** 2 ---> w = sqrt(area/ratio)
            area = scale ** 2
            anchor_wh[count, 0] = np.sqrt(area / ratio)
            anchor_wh[count, 1] = ratio * np.sqrt(area / ratio)
            count += 1
    anchors = np.zeros([h, w, 9, 4])
    for i in range(h):
        for j in range(w):
            anchor_xy = np.ones([9, 2])
            anchor_xy[:, 0] = anchor_xy[:, 0] * (j * 16)
            anchor_xy[:, 1] = anchor_xy[:, 1] * (i * 16)
            anchors[i, j, :, 2:4] = anchor_wh
            anchors[i, j, :, 0:2] = anchor_xy
    anchors = np.reshape(anchors, [-1, 4])
    return anchors

def resize_img_bbox(img, bboxes, IMG_W, IMG_H, channels, crop=False):
    if crop:
        img,bboxes = crop_center(img,bboxes,400,400)
    img_h, img_w = img.shape[0], img.shape[1]
    resized_bboxes = np.zeros_like(bboxes)
    resized_bboxes[:, 0] = IMG_W * bboxes[:, 0] / img_w
    resized_bboxes[:, 1] = IMG_H * bboxes[:, 1] / img_h
    resized_bboxes[:, 2] = IMG_W * bboxes[:, 2] / img_w
    resized_bboxes[:, 3] = IMG_H * bboxes[:, 3] / img_h
    resized_img = scm.imresize(img, [IMG_H, IMG_W, channels])
    resized_img = np.reshape(resized_img, [IMG_H,IMG_W,channels])
    #np.array(Image.fromarray(img).resize([IMG_W, IMG_H]))
    return resized_img, resized_bboxes

def pre_process(img):
    img = img / 255.0 
    img = img - 0.5
    img = img * 2.0
    return img

def cal_ious(anchors, gtbboxes):
    anchors = anchors[np.newaxis, :, :]
    gtbboxes = gtbboxes[:, np.newaxis, :]
    anchors_x1 = anchors[:, :, 0] - anchors[:, :, 2] / 2
    anchors_x2 = anchors[:, :, 0] + anchors[:, :, 2] / 2
    anchors_y1 = anchors[:, :, 1] - anchors[:, :, 3] / 2
    anchors_y2 = anchors[:, :, 1] + anchors[:, :, 3] / 2
    gtbboxes_x1 = gtbboxes[:, :, 0] - gtbboxes[:, :, 2] / 2
    gtbboxes_x2 = gtbboxes[:, :, 0] + gtbboxes[:, :, 2] / 2
    gtbboxes_y1 = gtbboxes[:, :, 1] - gtbboxes[:, :, 3] / 2
    gtbboxes_y2 = gtbboxes[:, :, 1] + gtbboxes[:, :, 3] / 2
    inter_x1 = np.maximum(anchors_x1, gtbboxes_x1)
    inter_x2 = np.minimum(anchors_x2, gtbboxes_x2)
    inter_y1 = np.maximum(anchors_y1, gtbboxes_y1)
    inter_y2 = np.minimum(anchors_y2, gtbboxes_y2)
    inter_area = np.maximum(0., inter_x2 - inter_x1) * np.maximum(0., inter_y2 - inter_y1)
    union_area = anchors[:, :, 2] * anchors[:, :, 3] + gtbboxes[:, :, 2] * gtbboxes[:, :, 3] - inter_area
    ious = inter_area / union_area
    return ious

def generate_minibatch(anchors, gtbboxes, IMG_W, IMG_H, MINIBATCH):
    #gtbboxes: [None, 4]
    nums = anchors.shape[0]
    anchors_x1 = anchors[:, 0] - anchors[:, 2]/2
    anchors_x2 = anchors[:, 0] + anchors[:, 2]/2
    anchors_y1 = anchors[:, 1] - anchors[:, 3]/2
    anchors_y2 = anchors[:, 1] + anchors[:, 3]/2
    illegal_idx0 = np.union1d(np.where(anchors_x1<0)[0], np.where(anchors_x2>=IMG_W)[0])
    illegal_idx1 = np.union1d(np.where(anchors_y1<0)[0], np.where(anchors_y2>=IMG_H)[0])
    illegal_idx = np.union1d(illegal_idx0, illegal_idx1)
    legal_idx = np.setdiff1d(np.array(range(nums)), illegal_idx)
    legal_anchors = anchors[legal_idx]
    ious = cal_ious(legal_anchors, gtbboxes)#[nums_obj, nums_anchor]
    max_iou_idx = np.where(np.abs(ious - np.max(ious, axis=1, keepdims=True)) < 1e-3)[1]
    ious = np.max(ious, axis=0)
    iou_greater_7_idx = np.where(ious >= 0.7)[0]
    pos_idx = np.union1d(max_iou_idx, iou_greater_7_idx)
    neg_idx = np.where(ious < 0.3)[0]
    neg_idx = np.setdiff1d(neg_idx, max_iou_idx)#remove some bboxes that may be iou < 0.3, but they are the maxest overlapping

    pos_nums = pos_idx.shape[0]
    neg_nums = neg_idx.shape[0]
    if neg_nums == 0:
        neg_idx = np.where(ious < 0.3)[0]
        neg_nums = neg_idx.shape[0]
        print('HERE....', neg_nums)
    #print("pos_nums: ", pos_nums, "neg_nums: ", neg_nums)
    if pos_nums < MINIBATCH//2:
        remain_nums = MINIBATCH - pos_nums
        rand_idx = np.random.randint(0, neg_nums, [remain_nums])
        neg_idx = neg_idx[rand_idx]
        batch_idx = np.concatenate((pos_idx, neg_idx), axis=0)
        batch_idx = legal_idx[batch_idx]
        labels = np.concatenate((np.ones([pos_nums]), np.zeros([remain_nums])))
        pos_anchor_bbox = legal_anchors[pos_idx]
        pos_iou = cal_ious(pos_anchor_bbox, gtbboxes)
        pos_gt_idx = np.argmax(pos_iou, axis=0)
        pos_gt_bbox = gtbboxes[pos_gt_idx]
    else:
        rand_pos_idx = np.random.randint(0, pos_nums, [MINIBATCH//2])
        try:
            rand_neg_idx = np.random.randint(0, neg_nums, [MINIBATCH//2])
        except:
            import pdb
            pdb.set_trace()
        batch_idx = np.concatenate((pos_idx[rand_pos_idx], neg_idx[rand_neg_idx]), axis=0)
        batch_idx = legal_idx[batch_idx]
        labels = np.concatenate((np.ones([MINIBATCH//2]), np.zeros([MINIBATCH//2])), axis=0)
        pos_anchor_bbox = legal_anchors[pos_idx[rand_pos_idx]]
        pos_iou = cal_ious(pos_anchor_bbox, gtbboxes)
        pos_gt_idx = np.argmax(pos_iou, axis=0)
        pos_gt_bbox = gtbboxes[pos_gt_idx]
    target_bbox = bbox2offset(pos_anchor_bbox, pos_gt_bbox)
    init_target_bbox = np.zeros([MINIBATCH, 4])

    init_target_bbox[:target_bbox.shape[0]] = target_bbox
    return batch_idx, labels, init_target_bbox

def bbox2offset(anchor_bbox, gt_bbox):
    t_x = (gt_bbox[:, 0:1] - anchor_bbox[:, 0:1])/anchor_bbox[:, 2:3]
    t_y = (gt_bbox[:, 1:2] - anchor_bbox[:, 1:2])/anchor_bbox[:, 3:4]
    t_w = np.log(gt_bbox[:, 2:3] / anchor_bbox[:, 2:3])
    t_h = np.log(gt_bbox[:, 3:4] / anchor_bbox[:, 3:4])
    return np.concatenate([t_x, t_y, t_w, t_h], axis=-1)

def pre_process_xml(XML_PATH, CLASSES):
    #print("............HERE............")
    xml_names = os.listdir(XML_PATH)
    #print("...........Initial number of xmls:", len(xml_names))
    final_xml = []
    for i in range(len(xml_names)):
        filename = os.path.join(XML_PATH, xml_names[i])
        try:
            tree = ET.parse(filename)
        except:
            import pdb
            pdb.set_trace()
        root = tree.getroot()
        objects = root.findall("object")
        names = []
        for idx, obj in enumerate(objects):
            curr_name = obj.find("name").text
            if curr_name in CLASSES:
                names.append(curr_name)
        if len(names)!=0:
            final_xml.append(xml_names[i])
    #print("...........Final number of xmls:", len(final_xml))
    return final_xml
        
def read_batch(anchors, XML_PATH, IMG_PATH, CLASSES, BATCHSIZE, IMG_H, IMG_W, channels, MINIBATCH, CROP, SUFFIX):    
    xml_names = pre_process_xml(XML_PATH, CLASSES)
    rand_idx = np.random.randint(0, len(xml_names), [BATCHSIZE])
    batch_imgs = np.zeros([BATCHSIZE, IMG_H, IMG_W, channels])
    batch_idxs = np.zeros([BATCHSIZE, MINIBATCH])
    masks = np.zeros([BATCHSIZE, MINIBATCH])
    target_bboxes = np.zeros([BATCHSIZE, MINIBATCH, 4])
    for i in range(BATCHSIZE):
        filename = xml_names[rand_idx[i]]
        img, gtbboxes, class_labels = read_data(XML_PATH + filename, IMG_PATH + filename[:-4] + SUFFIX, CLASSES)
        img, gtbboxes = resize_img_bbox(img, gtbboxes, IMG_W, IMG_H, channels, CROP)
        img = pre_process(img)
        #img = np.expand_dims(img, axis=-1)
        batch_idx, labels, target_bbox = generate_minibatch(anchors, gtbboxes,IMG_W, IMG_H, MINIBATCH)
        batch_idxs[i] = batch_idx
        masks[i] = labels
        target_bboxes[i] = target_bbox
        batch_imgs[i] = img
    return batch_imgs, batch_idxs, target_bboxes, masks