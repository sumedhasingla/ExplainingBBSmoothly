import sys
import os
import tensorflow as tf
import numpy as np
import scipy.io as sio
from PIL import Image
from Fast_RCNN.rpn_proposal.utils import  generate_anchors, pre_process_xml, draw_box
from Fast_RCNN.unified_network.networks import unified_net
from Fast_RCNN.unified_network.ops import offset2bbox, non_max_suppression

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
    '--config', '-c', default='configs/MIMIC_OD_Pacemaker.yaml')  
    parser.add_argument(
    '--main_dir', '-m', default='/pghbio/dbmi/batmanlab/singla/Explanation_XRay')  
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
    
    # ============= Experiment Parameters =============
    name = config['name']
    image_dir = config['image_dir']
    BATCHSIZE = 1 #config['BATCHSIZE']
    channels = config['num_channel']
    IMG_H = config['IMG_H'] 
    IMG_W = config['IMG_W'] 
    MINIBATCH = 256 #config['MINIBATCH'] 
    NUMS_PROPOSAL = 300 #config['NUMS_PROPOSAL'] 
    NMS_THRESHOLD = 0.7 #config['NMS_THRESHOLD'] 
    XML_PATH = config['XML_PATH']
    IMG_PATH = config['IMG_PATH']
    CLASSES = config['CLASSES']
    starting_step = config['starting_step']   
    # ============= Model =============
    anchors = generate_anchors(IMG_H, IMG_W)
    batch_size = 1
    imgs = tf.placeholder(tf.float32, [batch_size, IMG_H, IMG_W, 1])
    cls, reg, proposal = unified_net(imgs, anchors, CLASSES,  NUMS_PROPOSAL, NMS_THRESHOLD, IMG_H, IMG_W)

    x0, y0, x1, y1 = proposal[:, 0:1], proposal[:, 1:2], proposal[:, 2:3], proposal[:, 3:4]
    x, y, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    proposal = tf.concat([x, y, w, h], axis=1)
    normal_bbox, reverse_bbox = offset2bbox(reg, proposal)
    cls = tf.nn.softmax(cls)
    boxes, score, classes = non_max_suppression(cls, reverse_bbox, CLASSES)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    fast_rcnn_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_16") + \
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="classification") + \
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="regression")
    rpn_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rpn")
    saver = tf.train.Saver(fast_rcnn_var)
    saver.restore(sess, os.path.join(ckpt_dir,"model_frcnn_step6_"+name+".ckpt"))
    print("Fast RCNN Check point restored", os.path.join(ckpt_dir,"model_frcnn_step6_"+name+".ckpt"))
    print("Done.......")
    saver = tf.train.Saver(rpn_var)
    saver.restore(sess, os.path.join(ckpt_dir,"model_rpn_step4_"+name+".ckpt"))
    print("Model RPN Check point restored", os.path.join(ckpt_dir,"model_rpn_step4_"+name+".ckpt"))
    print("Done.......")
    
    image_names = os.listdir(IMG_PATH)
    for n in image_names:
        if '.jpg' in n:
            temp=np.array(Image.open(os.path.join(IMG_PATH, n)).resize([IMG_W, IMG_H]))             
            IMGS=temp
            break

    IMGS = np.asarray(IMGS)
    IMGS = np.expand_dims(IMGS, axis=-1)
    IMGS = IMGS[np.newaxis]
    print("IMGS: ",IMGS.shape)
    _cls = sess.run([cls, reg, proposal],feed_dict={imgs: IMGS})
    [BBOX, SCORE, CLS] = sess.run([boxes, score, classes], feed_dict={imgs: IMGS})
    print("score: ", SCORE)
    for i in range(SCORE.shape[0]):
        if SCORE[i] > 0.85:
            X0, Y0, X1, Y1 = BBOX[i, 0:1], BBOX[i, 1:2], BBOX[i, 2:3], BBOX[i, 3:4]
            X, Y, W, H = (X0 + X1) / 2, (Y0 + Y1) / 2, X1 - X0, Y1 - Y0
            BBOX1 = np.concatenate((X, Y, W, H), axis=-1)
            print(BBOX1)
            #img = draw_box(IMGS, np.asarray([BBOX1]), np.asarray([CLS[i]]), CLASSES, IMG_W, IMG_H)
            import scipy.misc
            scipy.misc.imsave('outfile'+str(i)+'.jpg', IMGS[0,:,:,0])
if __name__ == "__main__":
    Train()    