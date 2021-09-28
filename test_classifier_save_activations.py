import numpy as np
import pandas as pd
import sys 
import os
import pdb
import h5py
import yaml
import tensorflow as tf
from classifier.DenseNet import pretrained_classifier
from utils import read_data_file, load_images_and_labels, load_images
import tensorflow.contrib.slim as slim
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore")

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--config', '-c', default='configs/old/StanfordCheXpert_Classifier_256_ICLR.yaml')   
    parser.add_argument(
    '--main_dir', '-m', default='/pghbio/dbmi/batmanlab/singla/Explanation_XRay') 
    parser.add_argument(
    '--start', '-s', default=10000) 
    parser.add_argument(
    '--end', '-e', default=20000) 
    
    args = parser.parse_args()
    main_dir = args.main_dir
    start = int(args.start)
    end = int(args.end)
    # ============= Load config =============
    config_path = os.path.join(main_dir, args.config)
    config = yaml.load(open(config_path))
    print(config)
    # ============= Experiment Folder=============
    output_dir = '/pghbio/dbmi/batmanlab/singla/Image_Text_Project/MIMICCX/Data/512'
    ckpt_dir_continue = config['ckpt_dir_continue']
    if ckpt_dir_continue == '':
        past_checkpoint = output_dir
    else:
        past_checkpoint = os.path.join(main_dir, config['ckpt_dir_continue']  ) 
    # ============= Experiment Parameters =============
    BATCH_SIZE = config['batch_size']
    channels = config['num_channel']
    input_size = config['input_size'] 
    N_CLASSES = config['num_class']  
    file_names = config['names']
    # ============= Data =============
    data_dir = '/pghbio/dbmi/batmanlab/singla/Image_Text_Project/MIMICCX/CSV/APorPA'
    df_mimic = pd.read_csv(os.path.join(data_dir, 'cxr-dicom-lateral-labels-512-new-old-labels.csv'))
    df_mimic = df_mimic.fillna(2)
    print(df_mimic.shape)

    # ============= placeholder =============
    with tf.name_scope('input'):
        x_ = tf.placeholder(tf.float32, [None, input_size, input_size, channels], name='x-input')
    # ============= Model =============
    layers, prediction = pretrained_classifier(x_, n_label=N_CLASSES, reuse=False,  name='classifier', isTrain =False)
    # ============= Variables =============
    # Note that this list of variables only include the weights and biases in the model.
    lst_vars = []
    for v in tf.global_variables():
        lst_vars.append(v) 
    # ============= Session =============
    sess=tf.InteractiveSession()
    saver = tf.train.Saver(var_list=lst_vars)
    tf.global_variables_initializer().run()
    # ============= Load Checkpoint =============
    if past_checkpoint is not None:
        print("Before training, Load checkpoint ")
        print("Reading checkpoint...")
        class_vars = [var for var in slim.get_variables_to_restore() if 'classifier' in var.name]
        name_to_var_map_local = {var.op.name: var for var in class_vars}               
        temp_saver = tf.train.Saver(var_list=name_to_var_map_local)
        ckpt = tf.train.get_checkpoint_state(past_checkpoint+'/')
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            temp_saver.restore(sess, tf.train.latest_checkpoint(past_checkpoint+'/'))
            print("Classifier checkpoint loaded.................")
            print(past_checkpoint, ckpt_name)
        else:
            print("Failed checkpoint load")
            sys.exit()
    else:
        sys.exit()
    # ============= Testing Save the Output =============
    for index in range(start,end):
        row = df_mimic.iloc[index]
        temp = row['lateral']
        xs = load_images(np.asarray([temp]), '', input_size, channels, do_center_crop=config['do_center_crop'])
        [_layers, _prediction] = sess.run([layers, prediction], feed_dict={x_:xs}) 
        temp = temp.split('/')
        file_name = os.path.join(output_dir, temp[9], temp[10], temp[11].split('.')[0] + '_activation.hdf5')
        hf = h5py.File(file_name, 'w')
        for k in _layers.keys():
            hf.create_dataset(k, data=_layers[k])
        hf.close()

    
    
if __name__ == "__main__":
    test()