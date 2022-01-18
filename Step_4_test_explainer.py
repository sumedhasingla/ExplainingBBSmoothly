import sys 
import os
from classifier.DenseNet import pretrained_classifier
from unet.SegmentationNetwork import UNET
from explainer.networks import Discriminator, Discriminator_Ordinal, Generator_Encoder_Decoder
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
from utils import *
from losses import *
import pdb
import yaml
import time
import scipy.io as sio
from datetime import datetime
import random
import warnings
import argparse
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def convert_ordinal_to_binary(y,n):
    y = np.asarray(y).astype(int)
    new_y = np.zeros([y.shape[0], n])
    new_y[:,0] = y
    for i in range(0,y.shape[0]):
        for j in range(1,y[i]+1):
            new_y[i,j] = 1
    return new_y
def Train():    
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--config', '-c', default='configs/Step_4_MIMIC_Explainer_256_Pleural_Effusion.yaml')  
    parser.add_argument(
    '--main_dir', '-m', default='/ocean/projects/asc170022p/singla/ExplainingBBSmoothly')  
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
    if config['test_dir'] == '':
         config['test_dir'] = test_dir
    else:
        config['test_dir'] = os.path.join(main_dir, config['test_dir'])   
    # ============= Experiment Parameters =============
    if config['feature']:
        config['feature_names'] = config['feature_names'].split(',')
    image_dir = config['image_dir']
    ckpt_dir_cls = os.path.join(main_dir,config['cls_experiment']   )
    ckpt_dir_unet = os.path.join(main_dir,config['unet_experiment']   )
    BATCH_SIZE = 4
    ckpt_dir_continue = config['ckpt_dir_continue'] 
    if ckpt_dir_continue == '':
        ckpt_dir_continue = ckpt_dir
    else:
        ckpt_dir_continue = os.path.join(main_dir, ckpt_dir_continue, 'ckpt_dir')
    # ============= Data =============
    if config['img_to_save'] != '':
        config['img_to_save'] = os.path.join(main_dir,config['img_to_save'])
        data = np.load(config['img_to_save'])
        print("data: ", data.shape)
    elif config['names_to_save'] != '':
        config['names_to_save'] = os.path.join(main_dir,config['names_to_save'])
        data = np.load(config['names_to_save'])
        print("data: ", data.shape, data[0:4])
    else:    
        try:
            categories, file_names_dict = read_data_file(os.path.join(main_dir,config['image_label_dict']))
        except:
            print("Problem in reading input data file : ", config['image_label_dict'])
            sys.exit()
        data = np.asarray(file_names_dict.keys())
        print("The classification categories are: ")
        print(categories)
        print('The size of the training set: ', data.shape[0])
    # ============= placeholder =============
    x_source = tf.placeholder(tf.float32, [None, config['input_size'], config['input_size'], config['num_channel']])
    train_phase = tf.placeholder(tf.bool)
    if config['discriminator_type'] == 'Discriminator_Ordinal':
        y_s = tf.placeholder(tf.int32, [None, config['num_bins']])  
        y_source = y_s[:,0]
        y_t = tf.placeholder(tf.int32, [None, config['num_bins']]) 
        y_target = y_t[:,0]
    elif config['discriminator_type'] == 'Discriminator':
        y_s = tf.placeholder(tf.int32, [None])
        y_t = tf.placeholder(tf.int32, [None])
        y_source = y_s
        y_target = y_t

    # ============= G & D =============    
    G = Generator_Encoder_Decoder("generator") # with conditional BN, SAGAN: SN here as well
    if config['discriminator_type'] == 'Discriminator_Ordinal':
        D = Discriminator_Ordinal("discriminator") #with SN and projection
    elif config['discriminator_type'] == 'Discriminator':
        D = Discriminator("discriminator") 
    U = UNET()
    
    real_source_logits = D(x_source, y_s, config['num_bins'], "NO_OPS")
    
    fake_target_img, fake_target_img_embedding = G(x_source, train_phase, y_target, config['num_bins'], config['num_channel'])
    fake_source_img, fake_source_img_embedding = G(fake_target_img, train_phase, y_source, config['num_bins'], config['num_channel'])
    fake_source_recons_img, x_source_img_embedding = G(x_source, train_phase, y_source, config['num_bins'], config['num_channel'])    
    fake_target_logits = D(fake_target_img, y_t, config['num_bins'], None)    
    
    # ============= pre-trained classifier =============      
    real_img_feature_vector,  real_img_cls_prediction = pretrained_classifier(x_source, config['num_class'], reuse=False, name='classifier', return_layers=True, isTrain =False)
    fake_img_feature_vector, fake_img_cls_prediction = pretrained_classifier(fake_target_img, config['num_class'], reuse=True, return_layers=True, isTrain =False)
    real_img_recons_feature_vector, real_img_recons_cls_prediction = pretrained_classifier(fake_source_img, config['num_class'], reuse=True, return_layers=True, isTrain =False)
    # ============= pre-trained segmentation =============    
    _, seg_x_source = U(x_source)
    _, seg_x_fake = U(fake_target_img) 
        
    # ============= session =============
    sess = tf.Session()
    tf.compat.v1.random.set_random_seed(config['seed'])
    np.random.seed(config['seed'])
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
        
    # ============= Checkpoints =============
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
        print(ckpt_dir_continue)
        sys.exit()
    # ============= load pre-trained classifier checkpoint =============
    try:
        class_vars = [var for var in slim.get_variables_to_restore() if 'classifier' in var.name]
        name_to_var_map_local = {var.op.name: var for var in class_vars}               
        temp_saver = tf.train.Saver(var_list=name_to_var_map_local)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir_cls) 
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        temp_saver.restore(sess, os.path.join(ckpt_dir_cls, ckpt_name))
        print("Classifier checkpoint loaded.................")
        print(ckpt_dir_cls, ckpt_name)
    except:
        print("Cannot Load the classifier")
        pass
    
    class_vars = [var for var in slim.get_variables_to_restore() if 'UNet' in var.name]
    name_to_var_map_local = {var.op.name: var for var in class_vars}               
    temp_saver = tf.train.Saver(var_list=name_to_var_map_local)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir_unet) 
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    temp_saver.restore(sess, os.path.join(ckpt_dir_unet, ckpt_name))
    print("Unet checkpoint loaded.................",ckpt_name, ckpt_dir_unet )
    
    # ============= Testing =============     
    real_feature = {}
    fake_feature = {}
    np.random.shuffle(data)
    for i in range(data.shape[0] // BATCH_SIZE):
        image_paths = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        if config['img_to_save'] != '':   
            img = image_paths
        elif config['names_to_save'] != '':
            img = load_images_and_labels(image_paths, 
                                        image_dir=image_dir,
                                        input_size=config['input_size'],
                                        crop_size=config['crop_size'],
                                        num_channel=config['num_channel'])
        else:
            img, labels = load_images_and_labels(image_paths, 
                                            image_dir=image_dir,
                                            input_size=config['input_size'],
                                            n_class=1,
                                            attr_list=file_names_dict,
                                            crop_size=config['crop_size'],
                                            num_channel=config['num_channel'])
        
        img_repeat = np.repeat(img, config['num_bins'], 0)
        target_labels = np.asarray([np.asarray(range(config['num_bins'])) for j in range(img.shape[0])])
        target_labels = target_labels.ravel()
        if config['discriminator_type'] == 'Discriminator_Ordinal':
            target_labels = convert_ordinal_to_binary(target_labels,config['num_bins'])
        
        fake_img_, real_pred_, fake_pred_, real_seg_, fake_seg_ = sess.run([fake_target_img, real_img_cls_prediction, fake_img_cls_prediction, seg_x_source, seg_x_fake], feed_dict={y_t: target_labels,  x_source: img_repeat, train_phase: False})
        
        if config['feature']:
            real_f, fake_f = sess.run([real_img_feature_vector, fake_img_feature_vector], feed_dict={y_t: target_labels,  x_source: img_repeat, train_phase: False})
        
        if i == 0:
            names = np.asarray(image_paths)
            real_img = img
            fake_img = fake_img_
            real_pred = real_pred_
            fake_pred = fake_pred_
            real_seg = real_seg_
            fake_seg = fake_seg_
            if config['feature']:
                for f in config['feature_names']:
                    real_feature[f] = real_f[f]
                    fake_feature[f] = fake_f[f]
        else:
            names = np.append(names, np.asarray(image_paths), axis = 0)
            real_img = np.append(real_img, img, axis = 0)
            fake_img = np.append(fake_img, fake_img_, axis = 0)
            real_pred = np.append(real_pred, real_pred_, axis = 0)
            fake_pred = np.append(fake_pred, fake_pred_, axis = 0)
            real_seg = np.append(real_seg, real_seg_, axis = 0)
            fake_seg = np.append(fake_seg, fake_seg_, axis = 0)
            if config['feature']:
                for f in config['feature_names']:
                    real_feature[f] = np.append(real_feature[f],
                                                np.asarray(real_f[f]),
                                                axis=0)
                    fake_feature[f] = np.append(fake_feature[f],
                                                np.asarray(fake_f[f]),
                                                axis=0)
        
        if real_img.shape[0] > config['count_to_save']:
            break
        if i % 40 == 0:
            print(real_img.shape)
            np.save(os.path.join(test_dir, 'names'+config['suffix']+'.npy'), 
                    names)
            np.save(os.path.join(test_dir, 'real_img'+config['suffix']+'.npy'), 
                    real_img)
            np.save(os.path.join(test_dir, 'fake_img'+config['suffix']+'.npy'), 
                    fake_img)
            np.save(os.path.join(test_dir, 'real_pred'+config['suffix']+'.npy'), 
                    real_pred)
            np.save(os.path.join(test_dir, 'fake_pred'+config['suffix']+'.npy'), 
                    fake_pred)
            np.save(os.path.join(test_dir, 'real_seg'+config['suffix']+'.npy'), 
                    real_seg)
            np.save(os.path.join(test_dir, 'fake_seg'+config['suffix']+'.npy'), 
                    fake_seg)
            if config['feature']:
                for f in config['feature_names']:
                    np.save(os.path.join(test_dir,
                                         'real_'+f+config['suffix']+'.npy'),
                            real_feature[f])
                    np.save(os.path.join(test_dir,
                                         'fake_'+f+config['suffix']+'.npy'),
                            fake_feature[f])
            
    np.save(os.path.join(test_dir, 'names'+config['suffix']+'.npy'), 
            names)
    np.save(os.path.join(test_dir, 'real_img'+config['suffix']+'.npy'), 
            real_img)
    np.save(os.path.join(test_dir, 'fake_img'+config['suffix']+'.npy'), 
            fake_img)
    np.save(os.path.join(test_dir, 'real_pred'+config['suffix']+'.npy'), 
            real_pred)
    np.save(os.path.join(test_dir, 'fake_pred'+config['suffix']+'.npy'), 
            fake_pred)
    np.save(os.path.join(test_dir, 'real_seg'+config['suffix']+'.npy'), 
            real_seg)
    np.save(os.path.join(test_dir, 'fake_seg'+config['suffix']+'.npy'), 
            fake_seg)
    if config['feature']:
        for f in config['feature_names']:
            np.save(os.path.join(test_dir,
                                 'real_'+f+config['suffix']+'.npy'),
                    real_feature[f])
            np.save(os.path.join(test_dir,
                                 'fake_'+f+config['suffix']+'.npy'),
                    fake_feature[f])
        

if __name__ == "__main__":
    Train()