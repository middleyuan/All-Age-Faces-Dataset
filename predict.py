import tensorflow as tf
import numpy as np
from dlibdetect import FaceDetectorDlib
from tensorflow.contrib.layers import *
import os
import itertools
from question import Person, People, QuestionPosing
import csv

RESIZE_FINAL = 227
RESIZE_AOI = 256
GENDER_LIST =['F','M']
# AGE_LIST = ['[0, 3]','[4, 7]','[8, 14]','[15, 24]','[25, 37]','[38, 47]','[48, 59]','[60, 100]']
AGE_LIST = ['[0, 28]','[29, 54]','[55, 80]']
MAX_BATCH_SIZE = 128

tf.app.flags.DEFINE_string('gender_model_dir', './0.914_gender_checkpoint', 'Model directory (where checkpoint data lives)')
tf.app.flags.DEFINE_string('age_model_dir', './0.726_age_checkpoint', 'Model directory (where checkpoint data lives)')
tf.app.flags.DEFINE_string('device_id', '/cpu:0', 'What processing unit to execute inference on')
tf.app.flags.DEFINE_string('filename', '', 'File (Image) to process')

FLAGS = tf.app.flags.FLAGS

class ImageCoder(object):
    """Reference from rude-carnie"""
    def __init__(self):
        # Create a single Session to run all image coding calls.
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        self._sess = tf.compat.v1.Session(config=config)
        
        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.compat.v1.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
        
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.compat.v1.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        self.crop = tf.image.resize(self._decode_jpeg, (RESIZE_AOI, RESIZE_AOI))

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})
        
    def decode_jpeg(self, image_data):
        image = self._sess.run(self.crop, #self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})

        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def levi_hassner_bn(nlabels, images, pkeep, is_training):
    """Reference from rude-carnie"""
    
    batch_norm_params = {
        "is_training": is_training,
        "trainable": True,
        # Decay for the moving averages.
        "decay": 0.9997,
        # Epsilon to prevent 0s in variance.
        "epsilon": 0.001,
        # Collection containing the moving mean and moving variance.
        "variables_collections": {
            "beta": None,
            "gamma": None,
            "moving_mean": ["moving_vars"],
            "moving_variance": ["moving_vars"],
        }
    }
    weight_decay = 0.0005
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

    with tf.compat.v1.variable_scope("LeviHassnerBN", "LeviHassnerBN", [images]) as scope:

        with tf.contrib.slim.arg_scope(
                [convolution2d, fully_connected],
                weights_regularizer=weights_regularizer,
                biases_initializer=tf.constant_initializer(1.),
                weights_initializer=tf.random_normal_initializer(stddev=0.005),
                trainable=True):
            with tf.contrib.slim.arg_scope(
                    [convolution2d],
                    weights_initializer=tf.random_normal_initializer(stddev=0.01),
                    normalizer_fn=batch_norm,
                    normalizer_params=batch_norm_params):

                conv1 = convolution2d(images, 96, [7,7], [4, 4], padding='VALID', biases_initializer=tf.constant_initializer(0.), scope='conv1')
                pool1 = max_pool2d(conv1, 3, 2, padding='VALID', scope='pool1')
                conv2 = convolution2d(pool1, 256, [5, 5], [1, 1], padding='SAME', scope='conv2') 
                pool2 = max_pool2d(conv2, 3, 2, padding='VALID', scope='pool2')
                conv3 = convolution2d(pool2, 384, [3, 3], [1, 1], padding='SAME', biases_initializer=tf.constant_initializer(0.), scope='conv3')
                pool3 = max_pool2d(conv3, 3, 2, padding='VALID', scope='pool3')
                # can use tf.contrib.layer.flatten
                flat = tf.reshape(pool3, [-1, 384*6*6], name='reshape')
                full1 = fully_connected(flat, 512, scope='full1')
                drop1 = tf.nn.dropout(full1, pkeep, name='drop1')
                full2 = fully_connected(drop1, 512, scope='full2')
                drop2 = tf.nn.dropout(full2, pkeep, name='drop2')

    with tf.compat.v1.variable_scope('output') as scope:
        
        weights = tf.Variable(tf.random.normal([512, nlabels], mean=0.0, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
        output = tf.add(tf.matmul(drop2, weights), biases, name=scope.name)

    return output

def get_checkpoint(checkpoint_path, requested_step=None, basename='checkpoint'):
    """Reference from rude-carnie"""

    if requested_step is not None:

        model_checkpoint_path = '%s/%s-%s' % (checkpoint_path, basename, requested_step)
        if os.path.exists(model_checkpoint_path) is None:
            print('No checkpoint file found at [%s]' % checkpoint_path)
            exit(-1)
            print(model_checkpoint_path)
        print(model_checkpoint_path)
        return model_checkpoint_path, requested_step

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        # Restore checkpoint as described in top of this program
        print(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

        return ckpt.model_checkpoint_path, global_step
    else:
        print('No checkpoint file found at [%s]' % checkpoint_path)
        exit(-1)

def find_files(filename):
    if os.path.exists(filename): 
        return filename
    for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
        candidate = filename + suffix
        if os.path.exists(candidate):
            return candidate
    return None

def make_multi_crop_batch(filename, coder):
    """Reference and modified from rude-carnie project"""

    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if '.png' in filename:
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)
    
    image = coder.decode_jpeg(image_data)

    crops = []
    print('Running multi-cropped image')
    h = image.shape[0]
    w = image.shape[1]
    hl = h - RESIZE_FINAL
    wl = w - RESIZE_FINAL

    crop = tf.image.resize(image, (RESIZE_FINAL, RESIZE_FINAL))
    crops.append(tf.image.per_image_standardization(crop))
    crops.append(tf.image.per_image_standardization(tf.image.flip_left_right(crop)))

    corners = [ (0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl/2), int(wl/2))]
    for corner in corners:
        ch, cw = corner
        cropped = tf.image.crop_to_bounding_box(image, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
        crops.append(tf.image.per_image_standardization(cropped))
        flipped = tf.image.per_image_standardization(tf.image.flip_left_right(cropped))
        crops.append(tf.image.per_image_standardization(flipped))

    image_batch = tf.stack(crops)
    return image_batch

def classify_one_multi_crop(sess, label_list, softmax_output, images, image_file, coder=ImageCoder()):
    """Reference and modified from rude-carnie proejct"""
    try:
        print('Running file %s' % image_file)
        image_batch = make_multi_crop_batch(image_file, coder)
        
        batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
        output = batch_results[0]
        batch_sz = batch_results.shape[0]
    
        for i in range(1, batch_sz):
            output = output + batch_results[i]
        
        output /= batch_sz
        best = np.argmax(output)
        best_choice = (label_list[best], output[best])
        print('Guess @ 1 %s, prob = %.2f' % best_choice)
    
        nlabels = len(label_list)
        if nlabels > 2:
            tmp = output[best]
            output[best] = 0
            second_best = np.argmax(output)
            print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))

            return best, tmp, second_best, output[second_best]

        return best, output[best], not best, 1.0
    except Exception as e:
        print(e)
        print('Failed to run image %s ' % image_file)

def main(argv=None):

    print('Using dlib face detector...')
    detector = FaceDetectorDlib('./shape_predictor_68_face_landmarks.dat')
    face_files, outfile = detector.run(FLAGS.filename)  

    if len(face_files) != 0:

        faces_locations = detector.locations
        name_list = FLAGS.filename.split('/')
        name_list = name_list[len(name_list) - 1].split('.')
        basename = name_list[len(name_list) - 2]
        output = [basename]
        column = ['filename']

        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        coder = ImageCoder()
        age_predictions, age_second_predictions = [], []
        age_top1_probs, age_top2_probs = [], []
        gender_predictions = []
        gender_top1_probs = []
        ratios = detector.ratios
        people = []

        # age prediction
        age_graph = tf.Graph()
        with tf.compat.v1.Session(config=config, graph=age_graph) as sess:

            with tf.device(FLAGS.device_id):
                
                label_list = AGE_LIST
                nlabels = len(label_list)

                images = tf.compat.v1.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
                logits = levi_hassner_bn(nlabels, images, 1, False)
                init = tf.compat.v1.global_variables_initializer()

                model_checkpoint_path, _ = get_checkpoint(FLAGS.age_model_dir, None, 'checkpoint')

                saver = tf.compat.v1.train.Saver()
                saver.restore(sess, model_checkpoint_path)

                softmax_output = tf.nn.softmax(logits)

                image_files = list(filter(lambda x: x is not None, [find_files(f) for f in face_files]))

                for imagefile in image_files:
                    best, best_prob, second, second_prob = classify_one_multi_crop(sess, label_list, softmax_output, images, imagefile, coder)  
                    age_predictions.append(best)
                    age_top1_probs.append(best_prob)
                    age_second_predictions.append(second)
                    age_top2_probs.append(second_prob)

        # gender prediction
        gender_graph = tf.Graph()
        with tf.compat.v1.Session(config=config, graph=gender_graph) as sess:

            with tf.device(FLAGS.device_id):

                label_list = GENDER_LIST
                nlabels = len(label_list)

                images = tf.compat.v1.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
                logits = levi_hassner_bn(nlabels, images, 1, False)
                init = tf.compat.v1.global_variables_initializer()

                model_checkpoint_path, _ = get_checkpoint(FLAGS.gender_model_dir, None, 'checkpoint')

                saver = tf.compat.v1.train.Saver()
                saver.restore(sess, model_checkpoint_path)

                softmax_output = tf.nn.softmax(logits)

                for imagefile in image_files:
                    best, best_prob, second_best, second_prob = classify_one_multi_crop(sess, label_list, softmax_output, images, imagefile, coder)      
                    gender_predictions.append(best)  
                    gender_top1_probs.append(best_prob)

        data = People(age_predictions, age_top1_probs, age_second_predictions, age_top2_probs, gender_predictions, gender_top1_probs, ratios, faces_locations)
        question_posing = QuestionPosing('./Question Template/people.csv', data)
        print(question_posing.ask())
        output += [question_posing.question_type[question_posing.type]]
        column += ['relation']

        count = 0
        for (age_prediction, gender_prediction, ratio, location) in zip(age_predictions, gender_predictions, ratios, faces_locations):    
            output += [age_prediction]
            output += [gender_prediction]
            output += [ratio]
            output += [location]
            count += 1
            column += ['age_prediction']
            column += ['gender_prediction']
            column += ['ratio']
            column += ['location']
            if count == 5:
                break
        if count < 5:
            for i in range(count, 5):
                output += [-1]
                output += [-1]
                output += [-1]
                output += [-1]
                column += ['age_prediction']
                column += ['gender_prediction']
                column += ['ratio']
                column += ['location']

        with open('./output/%s.csv' % basename, 'w', newline='') as csvfile:

            writer = csv.writer(csvfile)

            writer.writerow(column)
            writer.writerow(output)

if __name__ == '__main__':
    tf.compat.v1.app.run()









