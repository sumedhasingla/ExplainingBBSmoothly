import sys
import os
import tensorflow as tf
import numpy as np
import scipy.io as sio
#from PIL import Image
from Fast_RCNN.rpn_proposal.utils import read_batch, generate_anchors, pre_process_xml
from Fast_RCNN.rpn_proposal.vggnet import vgg_16
from Fast_RCNN.rpn_proposal.ops import smooth_l1, offset2bbox
from Fast_RCNN.rpn_proposal.networks import rpn

from Fast_RCNN.fast_rcnn.networks import network
from Fast_RCNN.fast_rcnn.ops import smooth_l1, xywh2x1y1x2y2
from Fast_RCNN.fast_rcnn.utils import read_batch as read_batch_rcnn
from utils import load_images
import pdb
import yaml
import random
import warnings
import time
import argparse
warnings.filterwarnings("ignore", category=DeprecationWarning) 
tf.set_random_seed(0)
np.random.seed(0)

def Train():    
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--config', '-c', default='configs/MIMIC_OD_Heart.yaml')  
    parser.add_argument(
    '--main_dir', '-m', default='/ocean/projects/asc170022p/singla/Explanation_XRay')  
    args = parser.parse_args()
    main_dir = args.main_dir
    # ============= Load config =============
    config_path = os.path.join(main_dir, args.config)
    config = yaml.load(open(config_path))
    print(config)
    
    # ============= Experiment Folder=============
    assets_dir = os.path.join(main_dir, config['log_dir'], config['name'])
    log_dir = os.path.join(assets_dir, 'log')
    ckpt_dir = os.path.join(assets_dir, 'ckpt_dir')
    sample_dir = os.path.join(assets_dir, 'sample')
    test_dir = os.path.join(assets_dir, 'test')
    # make directory if not exist
    try: os.makedirs(log_dir)
    except: pass
    try: os.makedirs(ckpt_dir)
    except: pass
    try: os.makedirs(sample_dir)
    except: pass
    try: os.makedirs(test_dir)
    except: pass
    
    # ============= Experiment Parameters =============
    name = config['name']
    image_dir = config['image_dir']
    BATCHSIZE = config['BATCHSIZE']
    channels = config['num_channel']
    IMG_H = config['IMG_H'] 
    IMG_W = config['IMG_W'] 
    MINIBATCH = config['MINIBATCH'] 
    NUMS_PROPOSAL = config['NUMS_PROPOSAL'] 
    NMS_THRESHOLD = config['NMS_THRESHOLD'] 
    XML_PATH = config['XML_PATH']
    IMG_PATH = config['IMG_PATH']
    CLASSES = config['CLASSES']
    CROP = config['CROP']
    SUFFIX = config['SUFFIX']
    starting_step = config['starting_step']
    # ============= Data =============
    xml_files = pre_process_xml(XML_PATH, CLASSES)
    print("The classification CLASSES are: ")
    print(CLASSES)
    print('The size of the training set: ', len(xml_files))
    fp = open(os.path.join(log_dir, 'setting.txt'), 'w')
    fp.write('config_file:'+str(config_path)+'\n')
    fp.close()
    
    # ============= Model =============
    with tf.device('/gpu:0'): 
        imgs = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, 1])
        bbox_indxs = tf.placeholder(tf.int32, [None, MINIBATCH])
        masks = tf.placeholder(tf.int32, [None, MINIBATCH])
        target_bboxes = tf.placeholder(tf.float32, [None, MINIBATCH, 4])
        learning_rate = tf.placeholder(tf.float32)

        vgg_logits = vgg_16(imgs)
        cls, reg = rpn(vgg_logits)
        cls_logits = tf.concat([tf.nn.embedding_lookup(cls[i], bbox_indxs[i])[tf.newaxis] for i in range(BATCHSIZE)], axis=0)
        reg_logits = tf.concat([tf.nn.embedding_lookup(reg[i], bbox_indxs[i])[tf.newaxis] for i in range(BATCHSIZE)], axis=0)
        
        one_hot = tf.one_hot(masks, 2)
        pos_nums = tf.reduce_sum(tf.cast(masks, dtype=tf.float32))
        loss_cls = tf.reduce_sum(-tf.log(tf.reduce_sum(tf.nn.softmax(cls_logits) * one_hot, axis=-1) + 1e-10)) / pos_nums
        loss_reg = tf.reduce_sum(tf.reduce_sum(smooth_l1(reg_logits, target_bboxes), axis=-1) * tf.cast(masks, dtype=tf.float32)) / pos_nums
        regular = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        total_loss = loss_cls + loss_reg + regular * 0.0005
    with tf.variable_scope("Opt"):
        Opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(total_loss)
    # ============= Session =============
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) 
    # ============= Load VGG =============
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_16"))
    saver.restore(sess, os.path.join(main_dir, config['vgg_checkpoint']))
    print("VGG16 checkpoint restored", config['vgg_checkpoint'])
    
    saver = tf.train.Saver()
    # ============= Generate Initial Anchors =============
    anchors = generate_anchors(IMG_H, IMG_W)
    if starting_step == 1:
        print("******************** STEP-1 *********************************************")
        # ============= Round-1 RPN =============
        for i in range(2000):
            BATCH_IMGS, BATCH_IDXS, TARGET_BBOXS, MASKS = read_batch(anchors, XML_PATH, IMG_PATH, CLASSES, BATCHSIZE, IMG_H, IMG_W, channels, MINIBATCH, CROP, SUFFIX) # (2, 256, 256, 1), (2, 64), (2, 64, 4), (2, 64)
            sess.run(Opt, feed_dict={imgs: BATCH_IMGS, bbox_indxs: BATCH_IDXS, masks: MASKS, target_bboxes: TARGET_BBOXS, learning_rate: 0.001})
            if i % 100 == 0:
                [LOSS_CLS, LOSS_REG, TOTAL_LOSS] = sess.run([loss_cls, loss_reg, total_loss],
                feed_dict={imgs: BATCH_IMGS, bbox_indxs: BATCH_IDXS, masks: MASKS, target_bboxes: TARGET_BBOXS})
                print("Iteration: %d, total_loss: %f, loss_cls: %f, loss_reg: %f" % (i, TOTAL_LOSS, LOSS_CLS, LOSS_REG))
            if i % 1000 == 0:
                saver.save(sess, os.path.join(ckpt_dir,"model_rpn_"+name+".ckpt"))
        saver.save(sess, os.path.join(ckpt_dir,"model_rpn_"+name+".ckpt"))
        starting_step = 2
    # ============= Update Proposal =============
    if starting_step == 2:
        print("******************** STEP-2 *********************************************")
        cls, reg = cls[0], reg[0]
        scores = tf.nn.softmax(cls)[:, 1]

        anchors = tf.constant(anchors, dtype=tf.float32)
        normal_bbox, reverse_bbox = offset2bbox(reg, anchors)
        nms_idxs = tf.image.non_max_suppression(reverse_bbox, scores, max_output_size=2000, iou_threshold=NMS_THRESHOLD)
        bboxes = tf.nn.embedding_lookup(normal_bbox, nms_idxs)[:NUMS_PROPOSAL]

        saver.restore(sess, os.path.join(ckpt_dir,"model_rpn_"+name+".ckpt"))
        print("RPN checkpoint restored!!!", os.path.join(ckpt_dir,"model_rpn_"+name+".ckpt"))
        proposal_data = {}
        for idx, filename in enumerate(xml_files):
            img = load_images(np.asarray([IMG_PATH + filename[:-4] + SUFFIX]), '', IMG_W, 1, do_center_crop=False)
            #img = np.array(Image.open(IMG_PATH + filename[:-3] + "jpg").resize([IMG_W, IMG_H]))
            #img = np.expand_dims(img, axis=-1)
            BBOX,_cls = sess.run([bboxes, scores], feed_dict={imgs: img})
            x, y = (BBOX[:, 0:1] + BBOX[:, 2:3]) / 2, (BBOX[:, 1:2] + BBOX[:, 3:4]) / 2
            w, h = BBOX[:, 2:3] - BBOX[:, 0:1], BBOX[:, 3:4] - BBOX[:, 1:2]
            BBOX = np.concatenate((x, y, w, h), axis=-1)
            proposal_data[filename] = BBOX
            print("Total: %d, Current: %d"%(len(xml_files), idx))
        sio.savemat(os.path.join(ckpt_dir,"proposal_step2_"+name+".mat"), proposal_data)
        print("Proposal Data Saved!!!", os.path.join(ckpt_dir,"proposal_step2_"+name+".mat"))
        starting_step = 3
        
    if starting_step == 3:
        print("******************** STEP-3 *********************************************")
        proposals = sio.loadmat(os.path.join(ckpt_dir,"proposal_step2_"+name+".mat"))
        tf.reset_default_graph()
        
        imgs = tf.placeholder(tf.float32, [BATCHSIZE, IMG_H, IMG_W, 1])
        batch_proposal = tf.placeholder(tf.float32, [BATCHSIZE * MINIBATCH, 4])
        target_bboxes = tf.placeholder(tf.float32, [BATCHSIZE * MINIBATCH, 4])
        target_bboxes_idx = tf.placeholder(tf.int32, [BATCHSIZE * MINIBATCH])#for roi pooling
        target_classes = tf.placeholder(tf.int32, [BATCHSIZE * MINIBATCH])
        masks = tf.placeholder(tf.float32, [BATCHSIZE * MINIBATCH])
        learning_rate = tf.placeholder(tf.float32)

        batch_proposal_ = xywh2x1y1x2y2(batch_proposal, IMG_H, IMG_W)#for roi pooling
        cls, reg = network(imgs, batch_proposal_, target_bboxes_idx, CLASSES)
        print(cls)

        one_hot = tf.one_hot(target_classes, len(CLASSES) + 1)
        pos_nums = tf.reduce_sum(tf.cast(masks, dtype=tf.float32))
        loss_cls = tf.reduce_sum(-tf.log(tf.reduce_sum(tf.nn.softmax(cls) * one_hot, axis=-1) + 1e-10)) / pos_nums
        loss_reg = tf.reduce_sum(tf.reduce_sum(smooth_l1(reg, target_bboxes), axis=-1) * masks) / pos_nums
        regular = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        total_loss = loss_cls + loss_reg + regular * 0.0005
        with tf.variable_scope("Opt"):
            Opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(total_loss)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_16"))
        saver.restore(sess, os.path.join(main_dir, config['vgg_checkpoint']))
        print("VGG16 checkpoint restored", config['vgg_checkpoint'])
        
        saver = tf.train.Saver()
        
        for i in range(2000):
            BATCH_IMGS, BATCH_PROPOSAL, TARGET_BBOXES, TARGET_BBOXES_IDX, TARGET_CLASSES, MASKS = read_batch_rcnn(proposals, CLASSES, xml_files, XML_PATH, IMG_PATH, BATCHSIZE, MINIBATCH, IMG_H, IMG_W, CROP, SUFFIX)
            _ = sess.run([Opt], feed_dict={imgs: BATCH_IMGS, batch_proposal: BATCH_PROPOSAL, masks: MASKS, target_bboxes: TARGET_BBOXES, target_bboxes_idx: TARGET_BBOXES_IDX,target_classes: TARGET_CLASSES, learning_rate: 0.001})
            if i % 100 == 0:
                [LOSS_CLS, LOSS_REG, TOTAL_LOSS] = sess.run([loss_cls, loss_reg, total_loss], feed_dict={imgs: BATCH_IMGS, batch_proposal: BATCH_PROPOSAL, masks: MASKS, target_bboxes: TARGET_BBOXES, target_bboxes_idx: TARGET_BBOXES_IDX,target_classes: TARGET_CLASSES, learning_rate: 0.001})
                print("Iteration: %d, total_loss: %f, loss_cls: %f, loss_reg: %f" % (i, TOTAL_LOSS, LOSS_CLS, LOSS_REG))
                #pdb.set_trace()
            if i % 1000 == 0:
                saver.save(sess, os.path.join(ckpt_dir,"model_frcnn_"+name+".ckpt"))
        saver.save(sess, os.path.join(ckpt_dir,"model_frcnn_"+name+".ckpt"))

        starting_step = 4
    
    if starting_step == 4:
        print("******************** STEP-4 *********************************************")
        tf.reset_default_graph()
        
        imgs = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, 1])
        bbox_indxs = tf.placeholder(tf.int32, [BATCHSIZE, MINIBATCH])
        masks = tf.placeholder(tf.int32, [BATCHSIZE, MINIBATCH])
        target_bboxes = tf.placeholder(tf.float32, [BATCHSIZE, MINIBATCH, 4])
        learning_rate = tf.placeholder(tf.float32)

        vgg_logits = vgg_16(imgs)
        cls, reg = rpn(vgg_logits)
        cls_logits = tf.concat([tf.nn.embedding_lookup(cls[i], bbox_indxs[i])[tf.newaxis] for i in range(BATCHSIZE)], axis=0)
        reg_logits = tf.concat([tf.nn.embedding_lookup(reg[i], bbox_indxs[i])[tf.newaxis] for i in range(BATCHSIZE)], axis=0)

        one_hot = tf.one_hot(masks, 2)
        pos_nums = tf.reduce_sum(tf.cast(masks, dtype=tf.float32))
        loss_cls = tf.reduce_sum(-tf.log(tf.reduce_sum(tf.nn.softmax(cls_logits) * one_hot, axis=-1) + 1e-10)) / pos_nums
        loss_reg = tf.reduce_sum(tf.reduce_sum(smooth_l1(reg_logits, target_bboxes), axis=-1) * tf.cast(masks, dtype=tf.float32)) / pos_nums
        regular = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        total_loss = loss_cls + loss_reg + regular * 0.0005
        trainable_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rpn")
        with tf.variable_scope("Opt"):
            Opt = tf.train.MomentumOptimizer(learning_rate, momentum= 0.9).minimize(total_loss, var_list=trainable_var)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_16"))
        saver.restore(sess, os.path.join(ckpt_dir,"model_frcnn_"+name+".ckpt"))
        print("step 4 load: ", os.path.join(ckpt_dir,"model_frcnn_"+name+".ckpt"))
        saver = tf.train.Saver()
        anchors = generate_anchors(IMG_H, IMG_W)
        
        for i in range(2000):
            s = time.time()
            BATCH_IMGS, BATCH_IDXS, TARGET_BBOXS, MASKS = read_batch(anchors, XML_PATH, IMG_PATH, CLASSES, BATCHSIZE, IMG_H, IMG_W, channels, MINIBATCH,  CROP, SUFFIX) 
            e = time.time()
            read_time = e - s
            s = time.time()
            sess.run(Opt, feed_dict={imgs: BATCH_IMGS, bbox_indxs: BATCH_IDXS, masks: MASKS, target_bboxes: TARGET_BBOXS, learning_rate: 0.001})
            e = time.time()
            update_time = e - s
            if i % 100 == 0:
                [LOSS_CLS, LOSS_REG, TOTAL_LOSS] = sess.run([loss_cls, loss_reg, total_loss],
                feed_dict={imgs: BATCH_IMGS, bbox_indxs: BATCH_IDXS, masks: MASKS, target_bboxes: TARGET_BBOXS})
                print("Iteration: %d, total_loss: %f, loss_cls: %f, loss_reg: %f, read_time: %f, update_time: %f" % (i, TOTAL_LOSS, LOSS_CLS, LOSS_REG, read_time, update_time))
            if i % 1000 == 0:
                saver.save(sess, os.path.join(ckpt_dir,"model_rpn_step4_"+name+".ckpt"))
        saver.save(sess, os.path.join(ckpt_dir,"model_rpn_step4_"+name+".ckpt"))
        
        starting_step = 5
        
    if starting_step == 5:
        print("********************  STEP-5  *********************************************")
        cls, reg = cls[0], reg[0]
        scores = tf.nn.softmax(cls)[:, 1]
        anchors = tf.constant(anchors, dtype=tf.float32)
        normal_bbox, reverse_bbox = offset2bbox(reg, anchors)
        nms_idxs = tf.image.non_max_suppression(reverse_bbox, scores, max_output_size=2000, iou_threshold=NMS_THRESHOLD)
        bboxes = tf.nn.embedding_lookup(normal_bbox, nms_idxs)[:NUMS_PROPOSAL]
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(ckpt_dir,"model_rpn_step4_"+name+".ckpt"))

        proposal_data = {}
        for idx, filename in enumerate(xml_files):
            img = load_images(np.asarray([IMG_PATH + filename[:-4] + SUFFIX]), '', IMG_W, 1, do_center_crop=False)
            #img = np.array(Image.open(IMG_PATH + filename[:-3] + "jpg").resize([IMG_W, IMG_H]))
            #img = np.expand_dims(img, axis=-1)
            BBOX = sess.run(bboxes, feed_dict={imgs: img})
            x, y = (BBOX[:, 0:1] + BBOX[:, 2:3]) / 2, (BBOX[:, 1:2] + BBOX[:, 3:4]) / 2
            w, h = BBOX[:, 2:3] - BBOX[:, 0:1], BBOX[:, 3:4] - BBOX[:, 1:2]
            BBOX = np.concatenate((x, y, w, h), axis=-1)
            proposal_data[filename] = BBOX
            print("Total: %d, Current: %d"%(len(xml_files), idx))
        sio.savemat(os.path.join(ckpt_dir,"proposal_step5_"+name+".mat"), proposal_data)
        print("Proposal Data Saved!!!", os.path.join(ckpt_dir,"proposal_step5_"+name+".mat"))
        starting_step = 6
        
    if starting_step == 6:
        print("********************STEP-6*********************************************")
        proposals = sio.loadmat(os.path.join(ckpt_dir,"proposal_step5_"+name+".mat"))
        tf.reset_default_graph()
        
        imgs = tf.placeholder(tf.float32, [BATCHSIZE, IMG_H, IMG_W, 1])
        batch_proposal = tf.placeholder(tf.float32, [BATCHSIZE * MINIBATCH, 4])
        target_bboxes = tf.placeholder(tf.float32, [BATCHSIZE * MINIBATCH, 4])
        target_bboxes_idx = tf.placeholder(tf.int32, [BATCHSIZE * MINIBATCH])#for roi pooling
        target_classes = tf.placeholder(tf.int32, [BATCHSIZE * MINIBATCH])
        masks = tf.placeholder(tf.float32, [BATCHSIZE * MINIBATCH])
        learning_rate = tf.placeholder(tf.float32)

        batch_proposal_ = xywh2x1y1x2y2(batch_proposal, IMG_H, IMG_W)#for roi pooling
        cls, reg = network(imgs, batch_proposal_, target_bboxes_idx, CLASSES)

        one_hot = tf.one_hot(target_classes, len(CLASSES) + 1)
        pos_nums = tf.reduce_sum(tf.cast(masks, dtype=tf.float32))
        loss_cls = tf.reduce_sum(-tf.log(tf.reduce_sum(tf.nn.softmax(cls) * one_hot, axis=-1) + 1e-10)) / pos_nums
        loss_reg = tf.reduce_sum(tf.reduce_sum(smooth_l1(reg, target_bboxes), axis=-1) * masks) / pos_nums
        regular = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        total_loss = loss_cls + loss_reg + regular * 0.0005
        trainable_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_16/fc") +\
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="classification") + \
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="regression")
        with tf.variable_scope("Opt"):
            Opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(total_loss, var_list=trainable_var)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(ckpt_dir,"model_frcnn_"+name+".ckpt"))
        print("step 6 load: ", os.path.join(ckpt_dir,"model_frcnn_"+name+".ckpt"))
        
        for i in range(2001):
            BATCH_IMGS, BATCH_PROPOSAL, TARGET_BBOXES, TARGET_BBOXES_IDX, TARGET_CLASSES, MASKS = read_batch_rcnn(proposals, CLASSES, xml_files, XML_PATH, IMG_PATH, BATCHSIZE, MINIBATCH, IMG_H, IMG_W, CROP, SUFFIX)
            sess.run(Opt, feed_dict={imgs: BATCH_IMGS, batch_proposal: BATCH_PROPOSAL, masks: MASKS, target_bboxes: TARGET_BBOXES, target_bboxes_idx: TARGET_BBOXES_IDX,target_classes: TARGET_CLASSES, learning_rate: 0.001})
            if i % 100 == 0:
                [LOSS_CLS, LOSS_REG, TOTAL_LOSS] = sess.run([loss_cls, loss_reg, total_loss], feed_dict={imgs: BATCH_IMGS, batch_proposal: BATCH_PROPOSAL, masks: MASKS,
                                     target_bboxes: TARGET_BBOXES, target_bboxes_idx: TARGET_BBOXES_IDX,target_classes: TARGET_CLASSES, learning_rate: 0.001})
                print("Iteration: %d, total_loss: %f, loss_cls: %f, loss_reg: %f" % (i, TOTAL_LOSS, LOSS_CLS, LOSS_REG))
            if i % 1000 == 0:
                saver.save(sess, os.path.join(ckpt_dir,"model_frcnn_step6_"+name+".ckpt"))
        saver.save(sess, os.path.join(ckpt_dir,"model_frcnn_step6_"+name+".ckpt"))
        
if __name__ == "__main__":
    Train()