import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import scipy.misc as scm

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

def read_data(xml_path, img_path, CLASSES):
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
    labels = np.zeros([len(objects)])
    for idx, name in enumerate(names):
        labels[idx] = CLASSES.index(name)
    return img, gtbboxes, labels

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


def generate_minibatch(proposal, gtbboxes, classes, IMG_W, IMG_H, MINIBATCH, CLASSES):
    #gtbboxes: [None, 4]
    proposal_x1 = proposal[:, 0] - proposal[:, 2]/2
    proposal_x2 = proposal[:, 0] + proposal[:, 2]/2
    proposal_y1 = proposal[:, 1] - proposal[:, 3]/2
    proposal_y2 = proposal[:, 1] + proposal[:, 3]/2
    proposal_x1[proposal_x1 < 0.] = 0
    proposal_x2[proposal_x2 >= IMG_W] = IMG_W - 1
    proposal_y1[proposal_y1 < 0.] = 0
    proposal_y2[proposal_y2 >= IMG_H] = IMG_H - 1
    x, y = (proposal_x1 + proposal_x2) / 2, (proposal_y1 + proposal_y2) / 2
    w, h = proposal_x2 - proposal_x1, proposal_y2 - proposal_y1
    proposal = np.stack((x, y, w, h), axis=1)
    ious = cal_ious(proposal, gtbboxes)#[nums_obj, nums_anchor]
    max_iou_idx = np.where(np.abs(ious - np.max(ious, axis=1, keepdims=True)) < 1e-3)[1]
    ious = np.max(ious, axis=0)
    iou_greater_5_idx = np.where(ious >= 0.5)[0]
    pos_idx = np.union1d(max_iou_idx, iou_greater_5_idx)
    neg_idx = np.where(ious < 0.5)[0]
    neg_idx_ = np.where(ious >= 0.1)[0]
    neg_idx = np.intersect1d(neg_idx, neg_idx_)
    neg_idx = np.setdiff1d(neg_idx, max_iou_idx)#remove some bboxes that may be iou < 0.1, but they are the maxest overlapping

    pos_nums = pos_idx.shape[0]
    neg_nums = neg_idx.shape[0]
    if neg_nums == 0:
        neg_idx = np.where(ious < 0.3)[0]
        neg_nums = neg_idx.shape[0]
        print('HERE....', neg_nums, pos_nums)
    if pos_nums < MINIBATCH//4:
        remain_nums = MINIBATCH - pos_nums
        rand_idx = np.random.randint(0, neg_nums, [remain_nums])
        mini_batch_pos = proposal[pos_idx]
        mini_batch_neg = proposal[neg_idx[rand_idx]]
        mini_batch = np.concatenate((mini_batch_pos, mini_batch_neg), axis=0)
        mask = np.concatenate((np.ones([pos_nums]), np.zeros([remain_nums])))
        pos_iou = cal_ious(mini_batch_pos, gtbboxes)
        pos_gt_idx = np.argmax(pos_iou, axis=0)
        pos_gt_bbox = gtbboxes[pos_gt_idx]
        pos_classes = classes[pos_gt_idx]
    else:
        rand_pos_idx = np.random.randint(0, pos_nums, [MINIBATCH//4])
        rand_neg_idx = np.random.randint(0, neg_nums, [MINIBATCH * 3//4])
        mini_batch_pos = proposal[pos_idx[rand_pos_idx]]
        mini_batch_neg = proposal[neg_idx[rand_neg_idx]]
        mini_batch = np.concatenate((mini_batch_pos, mini_batch_neg), axis=0)
        mask = np.concatenate((np.ones([MINIBATCH//4]), np.zeros([MINIBATCH * 3//4])), axis=0)
        pos_iou = cal_ious(mini_batch_pos, gtbboxes)
        pos_gt_idx = np.argmax(pos_iou, axis=0)
        pos_gt_bbox = gtbboxes[pos_gt_idx]
        pos_classes = classes[pos_gt_idx]
    target_bbox = bbox2offset(mini_batch_pos, pos_gt_bbox)
    init_target_bbox = np.zeros([MINIBATCH, 4])
    init_target_classes = np.ones([MINIBATCH]) * len(CLASSES)
    init_target_classes[:pos_classes.shape[0]] = pos_classes
    init_target_bbox[:target_bbox.shape[0]] = target_bbox
    return mini_batch, mask, init_target_bbox, init_target_classes

def offset2bbox_np(pred_t, anchor_idx, anchors):
    anchors = anchors[np.int32(anchor_idx)]
    pred_t = pred_t[:anchor_idx.shape[0]]
    pred_bbox_x = pred_t[:, 0:1] * anchors[:, 2:3] + anchors[:, 0:1]
    pred_bbox_y = pred_t[:, 1:2] * anchors[:, 3:4] + anchors[:, 1:2]
    pred_bbox_w = np.exp(pred_t[:, 2:3]) * anchors[:, 2:3]
    pred_bbox_h = np.exp(pred_t[:, 3:4]) * anchors[:, 3:4]
    return np.concatenate((pred_bbox_x, pred_bbox_y, pred_bbox_w, pred_bbox_h), axis=-1)

def bbox2offset(anchor_bbox, gt_bbox):
    t_x = (gt_bbox[:, 0:1] - anchor_bbox[:, 0:1])/anchor_bbox[:, 2:3]
    t_y = (gt_bbox[:, 1:2] - anchor_bbox[:, 1:2])/anchor_bbox[:, 3:4]
    t_w = np.log(gt_bbox[:, 2:3] / anchor_bbox[:, 2:3])
    t_h = np.log(gt_bbox[:, 3:4] / anchor_bbox[:, 3:4])
    return np.concatenate([t_x, t_y, t_w, t_h], axis=-1)

def pre_process_xml1(XML_PATH, ):
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
def read_batch(proposals, CLASSES, xml_names, XML_PATH, IMG_PATH, BATCHSIZE, MINIBATCH, IMG_H, IMG_W, CROP, SUFFIX):
    rand_idx = np.random.randint(0, len(xml_names), [BATCHSIZE])
    batch_imgs = np.zeros([BATCHSIZE, IMG_H, IMG_W, 1])
    batch_proposal = np.zeros([BATCHSIZE * MINIBATCH, 4])
    masks = np.zeros([BATCHSIZE * MINIBATCH])
    target_bboxes = np.zeros([BATCHSIZE * MINIBATCH, 4])
    target_bboxes_idx = np.zeros([BATCHSIZE * MINIBATCH])
    target_classes = np.zeros([BATCHSIZE * MINIBATCH])

    for i in range(BATCHSIZE):
        filename = xml_names[rand_idx[i]]
        img, gtbboxes, class_labels = read_data(XML_PATH + filename, IMG_PATH + filename[:-4] + SUFFIX, CLASSES)
        img, gtbboxes = resize_img_bbox(img, gtbboxes, IMG_W, IMG_H, 1, CROP)
        img = pre_process(img)
        #img = np.expand_dims(img, axis=-1)
        proposal = proposals[filename]
        mini_batch, mask, target_bbox, target_class = generate_minibatch(proposal, gtbboxes, class_labels, IMG_W, IMG_H, MINIBATCH, CLASSES)
        batch_proposal[i*MINIBATCH:i*MINIBATCH+MINIBATCH] = mini_batch
        masks[i*MINIBATCH:i*MINIBATCH+MINIBATCH] = mask
        target_bboxes[i*MINIBATCH:i*MINIBATCH+MINIBATCH] = target_bbox
        target_classes[i*MINIBATCH:i*MINIBATCH+MINIBATCH] = target_class
        target_bboxes_idx[i*MINIBATCH:i*MINIBATCH+MINIBATCH] = i * np.ones([MINIBATCH])
        batch_imgs[i] = img
    return batch_imgs, batch_proposal, target_bboxes, target_bboxes_idx, target_classes, masks