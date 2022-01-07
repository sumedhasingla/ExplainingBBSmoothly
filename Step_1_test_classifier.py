import numpy as np
import pandas as pd
import sys 
import os
sys.path.append(os.path.abspath("/ocean/projects/asc170022p/singla/ExplainingBBSmoothly"))
import pdb
import yaml
import tensorflow as tf
from classifier.DenseNet import pretrained_classifier
from utils import read_data_file, load_images_and_labels
import tensorflow.contrib.slim as slim
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore")

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--config', '-c', default='configs/Step_1_StanfordCheXpert_Classifier_256.yaml'
)   
    parser.add_argument(
    '--main_dir', '-m', default='/ocean/projects/asc170022p/singla/ExplainingBBSmoothly') 
    
    args = parser.parse_args()
    main_dir = args.main_dir
    # ============= Load config =============
    config_path = os.path.join(main_dir, args.config)
    config = yaml.load(open(config_path))
    print(config)
    # ============= Experiment Folder=============
    output_dir = os.path.join(main_dir, config['log_dir'], config['name'])
    classifier_output_path = os.path.join(output_dir, config['output_folder_name'])
    try: os.makedirs(classifier_output_path)
    except: pass
    ckpt_dir_continue = config['ckpt_dir_continue']
    if ckpt_dir_continue == '':
        past_checkpoint = output_dir
    else:
        past_checkpoint = os.path.join(main_dir, config['ckpt_dir_continue']  ) 
    # ============= Experiment Parameters =============  
    file_names = config['names']
    categories = config['categories']
    categories = categories.split(',')
    print("The classification categories are: ", categories)
    # ============= Data =============
    if file_names == '':
        if config['use_output_csv']:
            df = pd.read_csv(config['output_csv'])
            config['path_column'] = config['output_csv_names_column']
        elif config['partition'] == 'test':
            df = pd.read_csv(config['test'])
        else:
            df = pd.read_csv(config['train'])
        df = df.fillna(0)
        print('The size of the datset: ', df.shape)
        data_file_names = np.asarray(list(df[config['path_column']]))
        
        #Create a dictionary, Key: file_path Value: Label
        file_names_dict = {}
        for index, row in df.iterrows():
            label = []
            for c in categories:
                label.append(row[c])
            file_names_dict[row[config['path_column']]] = label
    else:
        data_file_names = np.load(file_names,allow_pickle=True)
        print('The size of the dataset: ', data_file_names.shape[0])

    # ============= placeholder =============
    with tf.name_scope('input'):
        x_ = tf.placeholder(tf.float32, [None, config['input_size'], config['input_size'], config['num_channel']], name='x-input')
    # ============= Model =============
    feature_vector,prediction = pretrained_classifier(x_, n_label=config['num_class'] , reuse=False,  name='classifier', isTrain =False, return_layers=True)
    # ============= Variables =============
    # Note that this list of variables only include the weights and biases in the model.
    lst_vars = []
    for v in tf.trainable_variables():
        lst_vars.append(v) 
    # ============= Session =============
    sess=tf.InteractiveSession()
    saver = tf.train.Saver()
    tf.compat.v1.random.set_random_seed(config['seed'])
    np.random.seed(config['seed'])
    tf.global_variables_initializer().run()
    # ============= Load Checkpoint =============
    if past_checkpoint is not None:
        print("Before training, Load checkpoint ")
        print("Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(past_checkpoint+'/')
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(past_checkpoint, ckpt_name))
            print("Classifier checkpoint loaded.................", os.path.join(past_checkpoint, ckpt_name))
        else:
            print("Failed checkpoint load")
            sys.exit()
    else:
        sys.exit()
    # ============= Testing Save the Output =============
    names = np.empty([0])
    prediction_y = np.empty([0])
    true_y = np.empty([0])
    feature = {}
    feature_names = config['feature_names']
    feature_names = feature_names.split(',')
    num_batch = int(data_file_names.shape[0]/config['batch_size'])
    for i in range(0, num_batch):
        start = i*config['batch_size']
        ns = data_file_names[start:start+config['batch_size']]
        if file_names == '':
            file_names_dict
            xs, ys = load_images_and_labels(ns, 
                                            image_dir=config['image_dir'],
                                            input_size=config['input_size'],
                                            n_class=config['num_class'],
                                            attr_list=file_names_dict,
                                            crop_size=config['crop_size'],
                                            num_channel=config['num_channel'],
                                            uncertain=config['uncertain_label'])
        else:
            xs = load_images_and_labels(ns, 
                                        image_dir=config['image_dir'],
                                        input_size=config['input_size'],
                                        crop_size=config['crop_size'],
                                        num_channel=config['num_channel'])
        [_pred, _fv] = sess.run([prediction, feature_vector], feed_dict={x_:xs}) 
        if i == 0:
            names = np.asarray(ns)
            prediction_y = np.asarray(_pred)
            if config['feature']:
                for f in feature_names:
                    feature[f] = _fv[f]
            if file_names == '':
                true_y = np.asarray(ys)
        else:
            names = np.append(names, np.asarray(ns), axis= 0)
            prediction_y = np.append(prediction_y, np.asarray(_pred), axis=0)
            if config['feature']:
                for f in feature_names:
                    feature[f] = np.append(feature[f], np.asarray(_fv[f]), axis=0)
            if file_names == '':
                true_y = np.append(true_y, np.asarray(ys), axis= 0)
        if i%100 == 0:
            np.save(classifier_output_path + '/name_'+config['partition']+'.npy', names)
            np.save(classifier_output_path + '/prediction_y_'+config['partition']+'.npy', prediction_y)
            if config['feature']:
                for f in feature_names:
                    np.save(classifier_output_path + '/feature_'+ f + '_' +config['partition']+'.npy', feature[f])
            if file_names == '':
                np.save(classifier_output_path + '/true_y_'+config['partition']+'.npy', true_y)
    
    np.save(classifier_output_path + '/name_'+config['partition']+'.npy', names)
    np.save(classifier_output_path + '/prediction_y_'+config['partition']+'.npy', prediction_y)
    if config['feature']:
        for f in feature_names:
            np.save(classifier_output_path + '/feature_'+ f + '_' +config['partition']+'.npy', feature[f])
    if file_names == '':
        np.save(classifier_output_path + '/true_y_'+config['partition']+'.npy', true_y)
        # calculate metrics
        import sklearn.metrics
        num_cls = true_y.shape[1]
        for c in range(0, num_cls):
            pred = np.ravel(prediction_y[:,c])
            y_pred =  np.ravel(np.asarray(prediction_y[:,c] > 0.5).astype(int))
            y_true = np.ravel(true_y[:,c])
            print(categories[c])
            print(np.unique(true_y[:,c], return_counts=True))
            try:
                auc = sklearn.metrics.roc_auc_score(y_true, pred)
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, pred)
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                print("AUC: ", auc)
                print("Best Threshold: ", thresholds[ix])
            except:
                pass
            recall = sklearn.metrics.recall_score(y_true, y_pred)
            print("Recall: ", recall)
            cm = sklearn.metrics.confusion_matrix(y_true, y_pred) 
            print("Confusion Matrix: ", cm)
            print("...................................")
            


if __name__ == "__main__":
    test()