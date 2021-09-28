import tensorflow as tf
import cyclegan_datasets
import model
import pdb


def _load_samples(csv_name, image_type):
    filename_queue = tf.train.string_input_producer(
        [csv_name])

    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string)]

    filename_i, filename_j= tf.decode_csv(
        csv_filename, record_defaults=record_defaults)

    file_contents_i = tf.read_file(filename_i)
    file_contents_j = tf.read_file(filename_j)
    if image_type == '.jpeg':
        image_decoded_A = tf.image.decode_jpeg(
            file_contents_i, channels=model.IMG_CHANNELS)
        image_decoded_B = tf.image.decode_jpeg(
            file_contents_j, channels=model.IMG_CHANNELS)
    elif image_type == '.png':
        image_decoded_A = tf.image.decode_png(
            file_contents_i, channels=model.IMG_CHANNELS, dtype=tf.uint8)
        image_decoded_B = tf.image.decode_png(
            file_contents_j, channels=model.IMG_CHANNELS, dtype=tf.uint8)
    else:
        filename_i += filename_i + '_error'
        filename_j += filename_j + '_error'
    return image_decoded_A, image_decoded_B, filename_i, filename_j


def load_data(dataset_name, image_size_before_crop,
              do_shuffle=True, do_flipping=False):
    """

    :param dataset_name: The name of the dataset.
    :param image_size_before_crop: Resize to this size before random cropping.
    :param do_shuffle: Shuffle switch.
    :param do_flipping: Flip switch.
    :return:
    """
    if dataset_name not in cyclegan_datasets.DATASET_TO_SIZES:
        raise ValueError('split name %s was not recognized.'
                         % dataset_name)

    csv_name = cyclegan_datasets.PATH_TO_CSV[dataset_name]
    print(csv_name)
    image_i, image_j, filename_i, filename_j = _load_samples(
        csv_name, cyclegan_datasets.DATASET_TO_IMAGETYPE[dataset_name])
    inputs = {
        'image_i': image_i,
        'image_j': image_j,
        'filename_i': filename_i,
        'filename_j': filename_j
    }
    # Preprocessing:
    #center crop
    
    inputs['image_i'] = tf.image.central_crop(
        inputs['image_i'], 400/512.0)
    inputs['image_j'] = tf.image.central_crop(
        inputs['image_j'], 400/512.0)

    inputs['image_i'] = tf.image.resize_images(
        inputs['image_i'], [image_size_before_crop, image_size_before_crop])
    inputs['image_j'] = tf.image.resize_images(
        inputs['image_j'], [image_size_before_crop, image_size_before_crop])
    inputs['image_i'] = tf.reshape(
        inputs['image_i'], [image_size_before_crop, image_size_before_crop,1])
    inputs['image_j'] = tf.reshape(
        inputs['image_j'], [image_size_before_crop, image_size_before_crop,1])
    inputs['image_i'] = tf.multiply(tf.subtract(tf.div(inputs['image_i'], 255.0), 0.5), 2.0)
    inputs['image_j'] = tf.multiply(tf.subtract(tf.div(inputs['image_j'], 255.0), 0.5), 2.0)
    
    

    # Batch
    
    if do_shuffle is True:
        inputs['images_i'], inputs['images_j'] = tf.train.shuffle_batch(
            [inputs['image_i'], inputs['image_j']], 1, 5000, 100, seed=1)
    else:
        inputs['images_i'], inputs['images_j'] = tf.train.batch(
            [inputs['image_i'], inputs['image_j']], 1)

    return inputs

