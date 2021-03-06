import sys 
import os
from unet.SegmentationNetwork import UNET
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import pandas as pd
from losses import *
from skimage.transform import resize
import pdb
import yaml
import time
import scipy.io as sio
from datetime import datetime
import random
import warnings
import argparse
import cv2 as cv
warnings.filterwarnings("ignore", category=DeprecationWarning) 
 
def Train():    
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--config', '-c', default='configs/JSRT_Segmentation_256.yaml')  
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
    ckpt_dir_continue = config['ckpt_dir_continue'] 
    if ckpt_dir_continue == '':
        continue_train = False
    else:
        ckpt_dir_continue = os.path.join(main_dir, ckpt_dir_continue, 'ckpt_dir')
        continue_train = True
    # ============= Data =============
    df = pd.read_csv(config['input_csv'], sep = ',')
    df = df[~df[config['mask_column']].isnull()]

    print('The size of the training set: ', df.shape[0])
    fp = open(os.path.join(log_dir, 'setting.txt'), 'w')
    fp.write('config_file:'+str(config_path)+'\n')
    fp.close()
    
    # ============= placeholder =============
    x_source = tf.placeholder(tf.float32, [config['batch_size'], config['input_size'], config['input_size'], config['num_channel']])
    x_mask = tf.placeholder(tf.float32, [config['batch_size'], config['input_size'], config['input_size'], config['num_channel']]) 
    U = UNET() 
    # ============= pre-trained segmentation =============    
    _, seg_x_source = U(x_source)     
    # ============= Loss =============
    loss1 = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x_mask, seg_x_source))
    loss2 = dice_loss_1(x_mask, seg_x_source)
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss1+loss2, var_list=U.var_list())

    # ============= summary =============
    real_img_sum = tf.summary.image('real_img', x_source)
    mask_img_sum = tf.summary.image('mask_img', x_mask)
    pred_mask_sum = tf.summary.image('seg_x_source', seg_x_source)
    loss1_sum = tf.summary.scalar('loss1', loss1)
    loss2_sum = tf.summary.scalar('loss2', loss2)
    g_sum = tf.summary.merge([real_img_sum, mask_img_sum, pred_mask_sum, loss1_sum, loss2_sum])

    # ============= session =============
    lst_vars = []
    for v in tf.global_variables():
        lst_vars.append(v) 
        
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    
    # ============= Checkpoints =============
    if continue_train :
        print(" [*] before training, Load checkpoint ")
        print(" [*] Reading checkpoint...")
        
        ckpt = tf.train.get_checkpoint_state(ckpt_dir_continue)   
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_dir_continue, ckpt_name))
            print(ckpt_dir_continue, ckpt_name)
            print("Successful checkpoint upload")
        else:
            print("Failed checkpoint load")
    else:
        print(" [!] before training, no need to Load ")

    # ============= Training =============
    counter = 1
    for e in range(1, config['epochs']+1):
        # Shuffle data
        df = df.sample(frac=1.0,random_state=234)
        df = df.sample(frac=1.0,random_state=1)
        df = df.sample(frac=1.0,random_state=23)   
        imgs = np.asarray(df[config['img_column']])
        masks = np.asarray(df[config['mask_column']])
        for i in range(df.shape[0] // config['batch_size']):
            image_paths = imgs[i*config['batch_size']:(i+1)*config['batch_size']]
            mask_paths = masks[i*config['batch_size']:(i+1)*config['batch_size']]
            batch_imgs= []
            batch_masks = []
            for j in range(len(image_paths)):
                img = np.load(os.path.join(config['image_dir'],image_paths[j]))
                img = resize(img, (config['input_size'], config['input_size']))
                batch_imgs.append(img)
                mask = np.load(os.path.join(config['image_dir'],mask_paths[j]))
                mask = resize(mask, (config['input_size'], config['input_size']))
                batch_masks.append(mask)
            batch_imgs = np.asarray(batch_imgs)
            batch_masks = np.asarray(batch_masks)
            _, _g_sum  = sess.run([optimizer, g_sum],  feed_dict={x_source: batch_imgs,  x_mask: batch_masks})
            writer.add_summary(_g_sum, counter)  
            counter+=1
            
        if e%10==0:
            saver.save(sess, ckpt_dir + "/model%2d.ckpt" % e)


if __name__ == "__main__":
    Train()