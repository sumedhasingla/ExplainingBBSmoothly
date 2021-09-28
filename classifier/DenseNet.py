import tensorflow as tf
import sys 
import os
sys.path.append("/ocean/projects/asc170022p/singla/Explanation_XRay/classifier")
from layer import *
from tensorflow.layers import flatten
import pdb

def pretrained_classifier(inputae, n_label, reuse,  name='classifier', isTrain = False, n_filters = 64, output_bias=None, return_layers=False):
    print("Classifier", isTrain)
    layers = {}
    if output_bias is not None:
        output_bias = tf.constant_initializer(output_bias)
    padw = 3
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
            
        print("inputae: ", inputae)
        pad_input = tf.pad(inputae, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT") 
        #print("pad_input: ", pad_input)
        reluC, _conv, _bn = conv_2d_BN_Relu(pad_input, n_filters, 7, 2, 'VALID',isTrain)
        #print("conv1: ", conv1)
        layers['conv1'] = _conv
        padw = 1
        pad_conv1 = tf.pad(reluC, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT") 
        #print("pad_conv1: ", pad_conv1)        
        pool1 = Max_Pooling(pad_conv1, pool_size=[3,3], stride=2)
        #print("pool1: ", pool1)
        # Block 1
        block1 =  dense_block(pool1, nb_layers=6, layer_name='dense_1', isTrain=isTrain, filters=n_filters)
        layers['dense_1'] = block1
        transition1 = transition_layer(block1, isTrain, scope='trans_1')
        layers['trans_1'] = transition1
        #print("block1 output: ", transition1)
        #print("............................")
        block2 =  dense_block(transition1, nb_layers=12, layer_name='dense_2', isTrain=isTrain, filters=n_filters)
        layers['dense_2'] = block2
        transition2 = transition_layer(block2, isTrain, scope='trans_2')
        layers['trans_2'] = transition2
        #print("block2 output: ", transition2)
        #print("............................")
        block3 = dense_block(transition2, nb_layers=24, layer_name='dense_3', isTrain=isTrain, filters=n_filters)
        layers['dense_3'] = block3
        transition3 = transition_layer(block3, isTrain, scope='trans_3')
        layers['trans_3'] = transition3
        #print("block3 output: ", transition3)
        #print("............................")
        block4 = dense_block(transition3, nb_layers=16, layer_name='dense_final', isTrain=isTrain, filters=n_filters)
        layers['dense_4'] = block4
        #print("block4 output: ", block4)
        #print("............................")
        bn = batch_norm(block4, is_training=isTrain, name='linear_batch')
        print("bn final: ", bn)
        rel = tf.nn.relu(bn)
        shape = rel.get_shape().as_list()
        #print(shape)
        gap = Average_pooling(rel, pool_size=[shape[1],shape[2]], stride=1)
        #gap = Average_pooling(rel, pool_size=[7,7], stride=1)
        #print("global avg pooling final: ", gap)
        flat = flatten(gap)
        layers['gap'] = flat
        #print("flat: ", flat)
        logit = tf.layers.dense(flat, units=n_label, bias_initializer= output_bias, name='linear')
        layers['logit'] = logit
        #print("logit: ", logit)
        if isTrain == False:
            print(isTrain)
            logit1 = tf.stop_gradient(logit) 
        prediction = tf.nn.sigmoid(logit)
        layers['prediction'] = prediction
        if return_layers:
            return layers, prediction
        else:
            return logit,  prediction

#Feature Extractor
def pretrained_classifier_FE(inputae,reuse, isTrain=False):
    name='classifier'
    n_filters = 64
    padw = 3
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
            
        print("inputae: ", inputae)
        pad_input = tf.pad(inputae, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT") 
        conv1, _conv, _bn = conv_2d_BN_Relu(pad_input, n_filters, 7, 2, 'VALID',isTrain)
        padw = 1
        pad_conv1 = tf.pad(conv1, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")         
        pool1 = Max_Pooling(pad_conv1, pool_size=[3,3], stride=2)
        # Block 1
        block1 =  dense_block(pool1, nb_layers=6, layer_name='dense_1', isTrain=isTrain, filters=n_filters)
        transition1 = transition_layer(block1, isTrain, scope='trans_1')
        block2 =  dense_block(transition1, nb_layers=12, layer_name='dense_2', isTrain=isTrain, filters=n_filters)
        transition2 = transition_layer(block2, isTrain, scope='trans_2')
        block3 = dense_block(transition2, nb_layers=24, layer_name='dense_3', isTrain=isTrain, filters=n_filters)
        transition3 = transition_layer(block3, isTrain, scope='trans_3')
        block4 = dense_block(transition3, nb_layers=16, layer_name='dense_final', isTrain=isTrain, filters=n_filters)
        bn = batch_norm(block4, is_training=isTrain, name='linear_batch')
        rel = tf.nn.relu(bn)
        shape = rel.get_shape().as_list()
        gap = Average_pooling(rel, pool_size=[shape[1],shape[2]], stride=1)
        flat = flatten(gap)
        return flat

def pretrained_classifier_Mediation_part1(inputae, n_label, reuse,  name='classifier', isTrain = False, n_filters = 64, output_bias=None):
    print("Classifier", isTrain)
    layers = {}
    if output_bias is not None:
        output_bias = tf.constant_initializer(output_bias)
    padw = 3
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
            
        print("inputae: ", inputae)
        pad_input = tf.pad(inputae, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT") 
        #print("pad_input: ", pad_input)
        reluC, _conv, _bn = conv_2d_BN_Relu(pad_input, n_filters, 7, 2, 'VALID',isTrain)
        #print("conv1: ", conv1)
        layers['conv1'] = _conv
        padw = 1
        pad_conv1 = tf.pad(reluC, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT") 
        #print("pad_conv1: ", pad_conv1)        
        pool1 = Max_Pooling(pad_conv1, pool_size=[3,3], stride=2)
        #print("pool1: ", pool1)
        # Block 1
        block1 =  dense_block(pool1, nb_layers=6, layer_name='dense_1', isTrain=isTrain, filters=n_filters)
        layers['dense_1'] = block1
        transition1 = transition_layer(block1, isTrain, scope='trans_1')
        layers['trans_1'] = transition1
        #print("block1 output: ", transition1)
        #print("............................")
        block2 =  dense_block(transition1, nb_layers=12, layer_name='dense_2', isTrain=isTrain, filters=n_filters)
        layers['dense_2'] = block2
        transition2 = transition_layer(block2, isTrain, scope='trans_2')
        layers['trans_2'] = transition2
        #print("block2 output: ", transition2)
        #print("............................")
        block3 = dense_block(transition2, nb_layers=24, layer_name='dense_3', isTrain=isTrain, filters=n_filters)
        layers['dense_3'] = block3 
        return block3
    
def pretrained_classifier_Mediation_part2(block3, n_label, reuse,  name='classifier', isTrain = False, n_filters = 64, output_bias=None):
        layers = {}
        with tf.variable_scope(name) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            transition3 = transition_layer(block3, isTrain, scope='trans_3')
            layers['trans_3'] = transition3
            #print("block3 output: ", transition3)
            #print("............................")
            block4 = dense_block(transition3, nb_layers=16, layer_name='dense_final', isTrain=isTrain, filters=n_filters)
            layers['dense_4'] = block4
            #print("block4 output: ", block4)
            #print("............................")
            bn = batch_norm(block4, is_training=isTrain, name='linear_batch')
            print("bn final: ", bn)
            rel = tf.nn.relu(bn)
            shape = rel.get_shape().as_list()
            #print(shape)
            gap = Average_pooling(rel, pool_size=[shape[1],shape[2]], stride=1)
            #gap = Average_pooling(rel, pool_size=[7,7], stride=1)
            #print("global avg pooling final: ", gap)
            flat = flatten(gap)
            layers['gap'] = flat
            #print("flat: ", flat)
            logit = tf.layers.dense(flat, units=n_label, bias_initializer= output_bias, name='linear')
            layers['logit'] = logit
            #print("logit: ", logit)
            if isTrain == False:
                print(isTrain)
                logit1 = tf.stop_gradient(logit) 
            prediction = tf.nn.sigmoid(logit)
            layers['prediction'] = prediction
        #pred_y = tf.argmax(prediction, 1)
        return layers,  prediction

