import numpy as np
import pandas as pd
import sys 
import os
sys.path.append(os.path.abspath("/ocean/projects/asc170022p/singla/Explanation_XRay"))
import pdb
import yaml
import tensorflow as tf
from classifier.DenseNet import pretrained_classifier
from utils import read_data_file, load_images_and_labels
from Fast_RCNN.rpn_proposal.vggnet import vgg_16_train
import tensorflow.contrib.slim as slim
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore")

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--config', '-c', default='configs/StanfordCheXpert_Classifier_256_ICLR.yaml')   
    parser.add_argument(
    '--main_dir', '-m', default='/ocean/projects/asc170022p/singla/Explanation_XRay')  
    args = parser.parse_args()
    main_dir = args.main_dir
    # ============= Load config =============
    config_path = os.path.join(main_dir, args.config)
    config = yaml.load(open(config_path))
    print(config)
    # ============= Experiment Folder=============
    output_dir = os.path.join(main_dir, config['log_dir'], config['name'])
    try: os.makedirs(output_dir)
    except: pass
    try: os.makedirs(os.path.join(output_dir, 'logs'))
    except: pass
    # ============= Experiment Parameters =============
    seed = config['seed']
    BATCH_SIZE = config['batch_size']
    EPOCHS = config['epochs']
    channels = config['num_channel']
    input_size = config['input_size'] 
    N_CLASSES = config['num_class'] 
    ckpt_dir_continue = config['ckpt_dir_continue']
    classifier_type = config['classifier_type']
    categories = config['categories']
    categories = categories.split(',')
    print("The classification categories are: ")
    print(categories)
    if ckpt_dir_continue == '':
        continue_train = False
    else:
        ckpt_dir_continue = os.path.join(main_dir, config['ckpt_dir_continue']  )  
        continue_train = True
    training_columns_to_repeat = config['training_columns_to_repeat']
    weights_in_batch = config['weights_in_batch']
    only_frontal = config['only_frontal']
    # ============= Data =============
    df = pd.read_csv(config['train'])
    df = df.fillna(0)
    print('The size of the training set: ', df.shape)
    df_valid = pd.read_csv(config['test'])
    df_valid = df_valid.fillna(0)
    print('The size of the testing set: ', df_valid.shape[0])
    
    training_columns_to_repeat = training_columns_to_repeat.split(',')
    for c in  training_columns_to_repeat:
        if c in df.columns:
            df_c = df.loc[df[c] == 1]
            df = df.append(df_c,ignore_index=True)
    
    if only_frontal == 1:
        df = df.loc[df['Frontal/Lateral'] =='Frontal']
        df_valid = df_valid.loc[df_valid['Frontal/Lateral'] =='Frontal']
        
    print('The size of the updated training set: ', df.shape)
    data_train = np.asarray(list(df[config['path_column']]))
    data_test = np.asarray(list(df_valid[config['path_column']]))
    
    #Create a dictionary, Key: file_path Value: Label
    file_names_dict = {}
    for index, row in df.iterrows():
        label = []
        for c in categories:
            label.append(row[c])
        file_names_dict[row[config['path_column']]] = label
    for index, row in df_valid.iterrows():
        label = []
        for c in categories:
            label.append(row[c])
        file_names_dict[row[config['path_column']]] = label

    # ============= placeholder =============
    with tf.name_scope('input'):
        x_ = tf.placeholder(tf.float32, [None, input_size, input_size, channels], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, N_CLASSES], name='y-input') 
        w = tf.placeholder(tf.float32, [N_CLASSES], name='weight')
        isTrain = tf.placeholder(tf.bool) 
    # ============= Model =============    
    if classifier_type == 'DenseNet':
        logit,prediction = pretrained_classifier(x_, n_label=N_CLASSES, reuse=False,  name='classifier', isTrain =isTrain)
    elif classifier_type == 'vgg16':
        logit, prediction = vgg_16_train(x_, num_classes=N_CLASSES, is_training=isTrain,  scope='vgg_16')
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=y_,  logits=logit, pos_weight=w)
    loss = tf.reduce_mean(cross_entropy) 
    # ============= Optimization functions =============    
    train_step = tf.compat.v1.train.AdamOptimizer(0.0001).minimize(loss)
    # ============= summary =============    
    total_loss = tf.summary.scalar('total_loss', loss)
    sum_train = tf.summary.merge([total_loss]) 
    # ============= Session =============
    sess=tf.InteractiveSession()
    saver = tf.train.Saver()
    tf.compat.v1.random.set_random_seed(seed)
    np.random.seed(seed)
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter(output_dir + '/train', sess.graph)  
    writer_test = tf.summary.FileWriter(output_dir + '/test', sess.graph)    
    # ============= Checkpoints =============
    if continue_train:
        print("Before training, Load checkpoint ")
        print("Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(ckpt_dir_continue)
        print(ckpt_dir_continue)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_dir_continue, ckpt_name))
            print("Classifier checkpoint loaded.................")
            print(ckpt_dir_continue, ckpt_name)
        else:
            print("Failed checkpoint load")
    # ============= Training =============
    train_loss = []
    test_loss = []
    itr_train = 0
    itr_test = 0
    for epoch in range(EPOCHS):  
        total_loss = 0.0
        perm = np.arange(data_train.shape[0])
        np.random.shuffle(perm)
        data_train = data_train[perm]  
        num_batch = int(data_train.shape[0]/BATCH_SIZE)
        for i in range(0, num_batch):
            start = i*BATCH_SIZE
            ns = data_train[start:start+BATCH_SIZE]
            xs, ys = load_images_and_labels(ns, config['image_dir'],N_CLASSES, file_names_dict, input_size, channels, do_center_crop=config['do_center_crop'], uncertain=1)
            import pdb
            pdb.set_trace()
            if weights_in_batch == 1:
                count = np.sum(ys,axis=0)
                weight = []
                for index in range(N_CLASSES):
                    if count[index] == 0:
                        weight.append(1)
                    else:
                        weight.append((BATCH_SIZE - count[index])/count[index])
                weight = np.asarray(weight)
            else:
                weight = np.ones([N_CLASSES])

            [_, _loss,summary_str] = sess.run([train_step, loss, sum_train], feed_dict={x_:xs, isTrain:True, y_: ys, w:weight}) 
            writer.add_summary(summary_str, itr_train)  
            itr_train+=1
            total_loss += _loss
            
        total_loss /= i
        print("Epoch: " + str( epoch) + " loss: " + str(total_loss) + '\n')
        train_loss.append(total_loss)

        total_loss = 0.0
        perm = np.arange(data_test.shape[0])
        np.random.shuffle(perm)
        data_test = data_test[perm]  
        num_batch = int(data_test.shape[0]/BATCH_SIZE)
        for i in range(0, num_batch):
            start = i*BATCH_SIZE
            ns = data_test[start:start+BATCH_SIZE]
            xs, ys = load_images_and_labels(ns, config['image_dir'],N_CLASSES, file_names_dict, input_size, channels, do_center_crop=config['do_center_crop'], uncertain = 1)
            weight = np.ones([N_CLASSES])
            [_loss, summary_str] = sess.run([loss, sum_train], feed_dict={x_:xs, isTrain:False, y_: ys, w:weight}) 
            writer_test.add_summary(summary_str, itr_test)
            itr_test+=1
            total_loss += _loss
        total_loss /= i
        print("Epoch: "+ str(epoch) + " Test loss: "+ str(total_loss) + '\n')
        test_loss.append(total_loss)
        
        checkpoint_name = os.path.join( output_dir, 'cp_epoch'+str(epoch)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)
        np.save( os.path.join( output_dir,'logs','train_loss.npy'), np.asarray(train_loss))
        np.save( os.path.join( output_dir,'logs','test_loss.npy'), np.asarray(test_loss))
    checkpoint_name = os.path.join( output_dir, 'cp_epoch'+str(epoch)+'.ckpt')
    save_path = saver.save(sess, checkpoint_name)
    np.save( os.path.join( output_dir,'logs','train_loss.npy'), np.asarray(train_loss))
    np.save( os.path.join( output_dir,'logs','test_loss.npy'), np.asarray(test_loss))

    
if __name__ == "__main__":
    train()