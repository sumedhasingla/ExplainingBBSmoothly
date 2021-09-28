import tensorflow as tf
import pdb
from tensorflow.layers import flatten
def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.cast(y_true,tf.float32) - tf.cast(y_pred, tf.float32)))

def l1_loss_weighted(y_true, y_pred, mask_true):
    masked_true = tf.multiply(y_true, mask_true)
    masked_pred = tf.multiply(y_pred, mask_true)
    #cast
    masked_true = tf.cast(masked_true,tf.float32)
    masked_pred = tf.cast(masked_pred, tf.float32)
    
    l1 = tf.abs( masked_true - masked_pred ) #[n,256,256,1]
    
    masked_mse = tf.reduce_sum(l1, axis=[1,2,3]) /tf.math.maximum(tf.reduce_sum(mask_true, axis=[1,2,3]), 1)
    return tf.reduce_mean(masked_mse)

def l1_loss_weighted_1(y_true, y_pred, mask_true):
    reverse_mask_true = 1 - mask_true
    
    masked_true = tf.multiply(y_true, mask_true)
    masked_pred = tf.multiply(y_pred, mask_true)
    masked_true = tf.cast(masked_true,tf.float32)
    masked_pred = tf.cast(masked_pred, tf.float32) 
    l1 = tf.abs( masked_true - masked_pred ) #[n,256,256,1]    
    masked_mse = tf.reduce_sum(l1, axis=[1,2,3]) /tf.math.maximum(tf.reduce_sum(mask_true, axis=[1,2,3]), 1)
    total_loss = tf.reduce_mean(masked_mse)
    
    masked_true = tf.multiply(y_true, reverse_mask_true)
    masked_pred = tf.multiply(y_pred, reverse_mask_true)
    masked_true = tf.cast(masked_true,tf.float32)
    masked_pred = tf.cast(masked_pred, tf.float32) 
    l1 = tf.abs( masked_true - masked_pred ) #[n,256,256,1]    
    masked_mse = tf.reduce_sum(l1, axis=[1,2,3]) /tf.math.maximum(tf.reduce_sum(mask_true, axis=[1,2,3]), 1)
    masked_mse = tf.reduce_mean(masked_mse)
    
    
    return total_loss + masked_mse

def l2_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(tf.abs(tf.cast(y_true,tf.float32) - tf.cast(y_pred, tf.float32))))

def smooth_loss(mat):
    return tf.reduce_sum(tf.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :])) + \
           tf.reduce_sum(tf.abs(mat[:, :-1, :, :] - mat[:, 1:, :, :]))
           
def classification_loss(logit, label):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))

    return loss


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true * y_true, axis=-1) + tf.reduce_sum(y_pred * y_pred , axis=-1)

    return tf.reduce_mean(1 - (numerator ) / (denominator ))

def dice_loss_1(y_true,y_pred):   
    smooth = 1
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_coeff = (intersection * 2 + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    print(dice_coeff)
    return 1 - dice_coeff



def generator_loss(loss_func, fake):
    fake_loss = 0
    if loss_func.__contains__('wgan') :
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if loss_func == 'gan' or loss_func == 'dragan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge' :
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss
    
def discriminator_loss(loss_func, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if loss_func == 'hinge' :
        real_loss = -tf.reduce_mean(tf.minimum(0., -1.0 + real))
        fake_loss = -tf.reduce_mean(tf.minimum(0., -1.0 - fake))

    loss = real_loss + fake_loss

    return loss