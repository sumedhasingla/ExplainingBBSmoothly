import sys 
import os
from classifier.DenseNet import pretrained_classifier
from unet.SegmentationNetwork import UNET
from explainer.networks_128 import Discriminator, Discriminator_Ordinal, Generator_Encoder_Decoder
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
    '--config', '-c', default='configs/MIMIC_Explainer_256_PE.yaml')  
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
    if config['test_dir'] !='':
        test_dir = config['test_dir'] #os.path.join(assets_dir, 'test')
    else:
        test_dir = os.path.join(assets_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
    
    # ============= Experiment Parameters =============
    image_dir = config['image_dir']
    ckpt_dir_cls = os.path.join(main_dir,config['cls_experiment']   )
    ckpt_dir_unet = os.path.join(main_dir,config['unet_experiment']   )
    BATCH_SIZE = 4
    EPOCHS = config['epochs']
    channels = config['num_channel']
    input_size = config['input_size'] 
    NUMS_CLASS_cls = config['num_class']   
    NUMS_CLASS = config['num_bins']
    target_class = config['target_class']
    lambda_GAN = config['lambda_GAN']
    lambda_cyc = config['lambda_cyc']
    lambda_cls = config['lambda_cls']  
    save_summary = int(config['save_summary'])
    ckpt_dir_continue = ckpt_dir
    discriminator_type = config['discriminator_type']
    cls_loss_type = config['cls_loss_type']
    count_to_save = config['count_to_save']
    names_to_save = config['names_to_save']
    img_to_save = config['img_to_save']
    suffix = config['suffix']
    # ============= Data =============
    if img_to_save != '':
        img_to_save = os.path.join(main_dir,config['img_to_save'])
        data = np.load(img_to_save)
        print("data: ", data.shape)
    elif names_to_save != '':
        names_to_save = os.path.join(main_dir,config['names_to_save'])
        data = np.load(names_to_save)
        print("data: ", data.shape, data[0:4])
    else:
        # ============= Data =============
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
    x_source = tf.placeholder(tf.float32, [None, input_size, input_size, channels])
    train_phase = tf.placeholder(tf.bool)
    if discriminator_type == 'Discriminator_Ordinal':
        y_s = tf.placeholder(tf.int32, [None, NUMS_CLASS])  
        y_source = y_s[:,0]
        y_t = tf.placeholder(tf.int32, [None, NUMS_CLASS]) 
        y_target = y_t[:,0]
    elif discriminator_type == 'Discriminator':
        y_s = tf.placeholder(tf.int32, [None])
        y_t = tf.placeholder(tf.int32, [None])
        y_source = y_s
        y_target = y_t
    
    # ============= G & D =============    
    G = Generator_Encoder_Decoder("generator") # with conditional BN, SAGAN: SN here as well
    if discriminator_type == 'Discriminator_Ordinal':
        D = Discriminator_Ordinal("discriminator") #with SN and projection
    elif discriminator_type == 'Discriminator':
        D = Discriminator("discriminator") 
    U = UNET()
    real_source_logits = D(x_source, y_s, NUMS_CLASS, "NO_OPS")
    
    fake_target_img, fake_target_img_embedding = G(x_source, train_phase, y_target, NUMS_CLASS, channels)
    fake_source_img, fake_source_img_embedding = G(fake_target_img, train_phase, y_source, NUMS_CLASS, channels)
    fake_source_recons_img, x_source_img_embedding = G(x_source, train_phase, y_source, NUMS_CLASS, channels)    
    fake_target_logits = D(fake_target_img, y_t, NUMS_CLASS, None)    
    
    # ============= pre-trained classifier =============      
    real_img_cls_logit_pretrained,  real_img_cls_prediction = pretrained_classifier(x_source, NUMS_CLASS_cls, reuse=False, name='classifier')
    fake_img_cls_logit_pretrained, fake_img_cls_prediction = pretrained_classifier(fake_target_img, NUMS_CLASS_cls, reuse=True)
    real_img_recons_cls_logit_pretrained, real_img_recons_cls_prediction = pretrained_classifier(fake_source_img, NUMS_CLASS_cls, reuse=True)
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
        sys.exist()

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
        pass
    class_vars = [var for var in slim.get_variables_to_restore() if 'UNet' in var.name]
    name_to_var_map_local = {var.op.name: var for var in class_vars}               
    temp_saver = tf.train.Saver(var_list=name_to_var_map_local)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir_unet) 
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    temp_saver.restore(sess, os.path.join(ckpt_dir_unet, ckpt_name))
    print("Unet checkpoint loaded.................",ckpt_name, ckpt_dir_unet )
    
    # ============= Testing =============
    try:
        real_img = np.load(os.path.join(test_dir + '/real_img'+suffix+'.npy'))
        fake_images = np.load(os.path.join(test_dir + '/fake_images'+suffix+'.npy'))
        embedding = np.load(os.path.join(test_dir + '/embedding'+suffix+'.npy'))
        s_embedding = np.load(os.path.join(test_dir + '/s_embedding'+suffix+'.npy'))
        recons = np.load(os.path.join(test_dir + '/recons'+suffix+'.npy'))
        real_pred = np.load(os.path.join(test_dir + '/real_pred'+suffix+'.npy'))
        fake_pred = np.load(os.path.join(test_dir + '/fake_pred'+suffix+'.npy'))
        recons_pred = np.load(os.path.join(test_dir + '/recons_pred'+suffix+'.npy'))
        names = np.load(os.path.join(test_dir + '/names'+suffix+'.npy'))  
        seg = np.load(os.path.join(test_dir + '/seg'+suffix+'.npy')) 
        seg_fake_imgs = np.load(os.path.join(test_dir + '/seg_fake_imgs'+suffix+'.npy'))
        as_of_now = names.shape[0]// BATCH_SIZE
    except:
        real_img = np.empty([0])
        real_feature = np.empty([0])
        fake_images = np.empty([0])
        embedding = np.empty([0])
        s_embedding = np.empty([0])
        recons = np.empty([0])
        real_pred = np.empty([0])
        fake_pred = np.empty([0])
        fake_feature = np.empty([0])
        recons_pred = np.empty([0])
        names = np.empty([0])  
        seg = np.empty([0]) 
        seg_fake_imgs = np.empty([0]) 
        as_of_now = 0
    if names_to_save == '':
        np.random.shuffle(data)   
        np.random.shuffle(data) 
        np.random.shuffle(data) 
        data = data[0:count_to_save]
    print("as_of_now: ", as_of_now)          
    for i in range(as_of_now, data.shape[0] // BATCH_SIZE):
        image_paths = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        if img_to_save != '':
            img = image_paths
        elif names_to_save == '':
            img, labels = load_images_and_labels(image_paths, image_dir,1, file_names_dict, input_size, channels, do_center_crop=config['do_center_crop'])
            labels = labels.ravel()
            labels = np.repeat(labels, NUMS_CLASS, 0)
            if discriminator_type == 'Discriminator_Ordinal':
                labels = convert_ordinal_to_binary(labels,NUMS_CLASS)
        else:
            img = load_images(image_paths, image_dir, input_size, channels, do_center_crop=config['do_center_crop'])
        
        img_repeat = np.repeat(img, NUMS_CLASS, 0)
        target_labels = np.asarray([np.asarray(range(NUMS_CLASS)) for j in range(img.shape[0])])
        target_labels = target_labels.ravel()
        if discriminator_type == 'Discriminator_Ordinal':
            target_labels = convert_ordinal_to_binary(target_labels,NUMS_CLASS)
        
        FAKE_IMG, f_embed, real_p, fake_p, seg_source, seg_fake , real_f, fake_f= sess.run([fake_target_img, fake_target_img_embedding, real_img_cls_prediction, fake_img_cls_prediction, seg_x_source, seg_x_fake, real_img_cls_logit_pretrained, fake_img_cls_logit_pretrained], feed_dict={y_t: target_labels,  x_source: img_repeat, train_phase: False})
        recons_img, recons_p, s_embed = np.zeros(FAKE_IMG.shape), np.zeros(fake_p.shape),  np.zeros(f_embed.shape)
        if names_to_save == '':
            recons_img, recons_p, s_embed = sess.run([fake_source_img, real_img_recons_cls_prediction,x_source_img_embedding], feed_dict={y_t: target_labels,  x_source: img_repeat, train_phase: False, y_s:labels})  
        
        if i == 0:
            real_img = img
            #real_feature = real_f
            #fake_feature= fake_f
            fake_images = FAKE_IMG
            embedding = f_embed
            s_embedding = s_embed
            recons = recons_img
            names = np.asarray(image_paths)
            real_pred = real_p
            fake_pred = fake_p
            recons_pred = recons_p  
            seg = seg_source
            seg_fake_imgs = seg_fake
        else:
            real_img = np.append(real_img, img, axis = 0)
            #real_feature = np.append(real_feature, real_f, axis = 0)
            #fake_feature= np.append(fake_feature, fake_f, axis = 0)
            fake_images =np.append(fake_images, FAKE_IMG, axis = 0)
            embedding = np.append(embedding, f_embed, axis = 0)
            s_embedding = np.append(s_embedding, f_embed, axis = 0)
            recons = np.append(recons, recons_img, axis = 0)
            names = np.append(names, np.asarray(image_paths), axis = 0)
            real_pred = np.append(real_pred, real_p, axis = 0)
            fake_pred = np.append(fake_pred, fake_p, axis = 0)
            recons_pred = np.append(recons_pred, recons_p, axis = 0)
            seg = np.append(seg, seg_source, axis = 0)
            seg_fake_imgs = np.append(seg_fake_imgs, seg_fake, axis = 0)
            
        if i % 40 == 0:
            print(real_img.shape)
            np.save(os.path.join(test_dir + '/real_img'+suffix+'.npy'),   real_img     )
            #np.save(os.path.join(test_dir + '/real_feature'+suffix+'.npy'),   real_feature     )
            #np.save(os.path.join(test_dir + '/fake_feature'+suffix+'.npy'),   fake_feature     )
            np.save(os.path.join(test_dir + '/fake_images'+suffix+'.npy'),   fake_images     )
            np.save(os.path.join(test_dir + '/embedding'+suffix+'.npy'),   embedding     )
            np.save(os.path.join(test_dir + '/s_embedding'+suffix+'.npy'),   s_embedding     )
            np.save(os.path.join(test_dir + '/recons'+suffix+'.npy'),   recons     )
            np.save(os.path.join(test_dir + '/names'+suffix+'.npy'),   names     )
            np.save(os.path.join(test_dir + '/real_pred'+suffix+'.npy'),   real_pred     )    
            np.save(os.path.join(test_dir + '/fake_pred'+suffix+'.npy'),   fake_pred     )    
            np.save(os.path.join(test_dir + '/recons_pred'+suffix+'.npy'),   recons_pred     )   
            np.save(os.path.join(test_dir + '/seg'+suffix+'.npy'),   seg     ) 
            np.save(os.path.join(test_dir + '/seg_fake_imgs'+suffix+'.npy'),   seg_fake_imgs     ) 

    np.save(os.path.join(test_dir + '/real_img'+suffix+'.npy'),   real_img     )
    #np.save(os.path.join(test_dir + '/real_feature'+suffix+'.npy'),   real_feature     )
    #np.save(os.path.join(test_dir + '/fake_feature'+suffix+'.npy'),   fake_feature     )
    np.save(os.path.join(test_dir + '/fake_images'+suffix+'.npy'),   fake_images     )
    np.save(os.path.join(test_dir + '/embedding'+suffix+'.npy'),   embedding     )
    np.save(os.path.join(test_dir + '/s_embedding'+suffix+'.npy'),   s_embedding     )
    np.save(os.path.join(test_dir + '/recons'+suffix+'.npy'),   recons     )
    np.save(os.path.join(test_dir + '/names'+suffix+'.npy'),   names     )
    np.save(os.path.join(test_dir + '/real_pred'+suffix+'.npy'),   real_pred     )    
    np.save(os.path.join(test_dir + '/fake_pred'+suffix+'.npy'),   fake_pred     )    
    np.save(os.path.join(test_dir + '/recons_pred'+suffix+'.npy'),   recons_pred     ) 
    np.save(os.path.join(test_dir + '/seg'+suffix+'.npy'),   seg     )  
    np.save(os.path.join(test_dir + '/seg_fake_imgs'+suffix+'.npy'),   seg_fake_imgs     ) 
        

if __name__ == "__main__":
    Train()