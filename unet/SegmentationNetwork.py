import tensorflow as tf

def upsampling(inputs):
    H = inputs.shape[1]
    W = inputs.shape[2]
    return tf.image.resize_nearest_neighbor(inputs, [H * 2, W * 2])

class UNET:
    def __init__(self, name='UNet'):
        self.name = name
        
    def __call__(self, x):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            merge_axis = -1 # Feature maps are concatenated along last axis (for tf backend)
            n_filter = 32
            k_size = 3
            print("x: ", x)
            conv1 = tf.layers.conv2d(inputs=x, filters=n_filter, kernel_size=k_size, padding='same', activation=tf.nn.relu)
            conv1 = tf.layers.conv2d(inputs=conv1, filters=n_filter, kernel_size=k_size, padding='same', activation=tf.nn.relu)
            print("conv1: ", conv1)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2, padding='same')
            print("pool1: ", pool1)
            conv2 = tf.layers.conv2d(inputs=pool1, filters=n_filter*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(inputs=conv2, filters=n_filter*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)
            print("conv2: ", conv2)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2, padding='same')
            print("pool2: ", pool2)
            conv3 = tf.layers.conv2d(inputs=pool2, filters=n_filter*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(inputs=conv3, filters=n_filter*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2, padding='same')

            conv4 = tf.layers.conv2d(inputs=pool3, filters=n_filter*2*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)
            output=conv4
            conv4 = tf.layers.conv2d(inputs=conv4, filters=n_filter*2*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=2, strides=2, padding='same')

            conv5 = tf.layers.conv2d(inputs=pool4, filters=n_filter*2*2*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)

            up1 = upsampling(conv5)

            conv6 = tf.layers.conv2d(inputs=up1, filters=n_filter*2*2*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)
            conv6 = tf.layers.conv2d(inputs=conv6, filters=n_filter*2*2*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)

            merged1 = tf.concat([conv4, conv6], axis=merge_axis)

            conv6 =  tf.layers.conv2d(inputs=merged1, filters=n_filter*2*2*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)

            up2 = upsampling(conv6)

            conv7 = tf.layers.conv2d(inputs=up2, filters=n_filter*2*2*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)
            conv7 = tf.layers.conv2d(inputs=conv7, filters=n_filter*2*2*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)

            merged2 = tf.concat([conv3, conv7], axis=merge_axis)
            conv7 = tf.layers.conv2d(inputs=merged2, filters=n_filter*2*2*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)

            up3 = upsampling(conv7)
            conv8 = tf.layers.conv2d(inputs=up3, filters=n_filter*2*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)
            conv8 = tf.layers.conv2d(inputs=conv8, filters=n_filter*2*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)

            merged3 = tf.concat([conv2, conv8], axis=merge_axis)
            conv8 = tf.layers.conv2d(inputs=merged3, filters=n_filter*2*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)

            up4 = upsampling(conv8)
            conv9 = tf.layers.conv2d(inputs=up4, filters=n_filter*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)
            conv9 = tf.layers.conv2d(inputs=conv9, filters=n_filter*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)
            merged4 =  tf.concat([conv1, conv9], axis=merge_axis)
            conv9 = tf.layers.conv2d(inputs=merged4, filters=n_filter*2, kernel_size=k_size, padding='same', activation=tf.nn.relu)

            conv10 = tf.layers.conv2d(inputs=conv9, filters=1, kernel_size=k_size, padding='same', activation=tf.nn.sigmoid)

            #output = conv10
            return output, conv10
        
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


