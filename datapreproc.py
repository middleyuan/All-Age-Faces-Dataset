from datetime import datetime
import os, sys 
import random
import threading
import numpy as np
import tensorflow as tf
import json

RESIZE_HEIGHT = 256
RESIZE_WIDTH =  256

tf.app.flags.DEFINE_string('list_dir', './All-Age-Faces Dataset/image sets', 'Label directory')
tf.app.flags.DEFINE_string('data_dir', './All-Age-Faces Dataset/aglined faces','Data directory')
tf.app.flags.DEFINE_string('output_age_dir', './age','Output age directory')
tf.app.flags.DEFINE_string('output_gender_dir', './gender','Output gender directory')
tf.app.flags.DEFINE_string('train_list', 'train_.txt','Training list')
tf.app.flags.DEFINE_string('valid_list', 'val_.txt','Test list')
tf.app.flags.DEFINE_integer('train_shards', 10,'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('valid_shards', 2,'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 2,'Number of threads to preprocess the images.')
FLAGS = tf.app.flags.FLAGS

class ImageHandler(object):

    def __init__(self):
        self._sess = tf.compat.v1.Session()

        self._png_data = tf.compat.v1.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        self._decode_jpeg_data = tf.compat.v1.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        cropped = tf.image.resize(self._decode_jpeg, [RESIZE_HEIGHT, RESIZE_WIDTH])
        cropped = tf.cast(cropped, tf.uint8) 
        self._recoded = tf.image.encode_jpeg(cropped, format='rgb', quality=100)

    def _is_png(self, filename):
        return '.png' in filename
    
    def get_image_data(self, filename):
        with tf.gfile.FastGFile(filename, 'rb') as f:
            self.image_data = f.read()

        if self._is_png(filename):
            self.image_data = self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: self.image_data})
        
        return self.image_data

    def resample_jpeg(self, image_data):
        return self._sess.run(self._recoded, feed_dict={self._decode_jpeg_data: image_data})

class DataProcessor(object):

    def __init__(self, train_filename, valid_filename, directory, train_shards, valid_shards):
        self.train_filename = train_filename
        self.valid_filename = valid_filename
        self.directory = directory
        self.train_agelabels = []
        self.train_genderlabels = []
        self.val_agelabels = []
        self.val_genderlabels = []
        self.train_filenames = []
        self.valid_filenames = []
        self.train_shards = train_shards
        self.valid_shards = valid_shards
        self.handler = ImageHandler()

    def process_dataset(self):

        # training data
        self.find_image_files(self.train_filename, True)
        self.process_image_files('train', self.train_filenames, self.train_agelabels, self.train_shards)
        self.process_image_files('train', self.train_filenames, self.train_genderlabels, self.train_shards)

        # valid data
        self.find_image_files(self.valid_filename, False)  
        self.process_image_files('validation', self.valid_filenames, self.val_agelabels, self.valid_shards)
        self.process_image_files('validation', self.valid_filenames, self.val_genderlabels, self.valid_shards)

    def int64_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def return_example(self, filename, image, label, height, width):
        return tf.train.Example(features=tf.train.Features(feature={  'image/class/label': self.int64_feature(label),
                                                                    'image/filename': self.bytes_feature(str.encode(os.path.basename(filename))),
                                                                    'image/encoded': self.bytes_feature(image),
                                                                    'image/height': self.int64_feature(height),
                                                                    'image/width': self.int64_feature(width)}))

    def find_image_files(self, list_file, istrain):

        files_genderlabels = [l.strip().split(' ') for l in tf.gfile.FastGFile(list_file, 'r').readlines()]

        genderlabels, agelabels, filenames = [], [], []
        
        for path, genderlabel in files_genderlabels:
            jpeg_file_path = '%s/%s' % (self.directory, path)
            if os.path.exists(jpeg_file_path):
                genderlabels.append(genderlabel)
                file_labels = path.strip().split('.')
                file_labels = file_labels[0].strip().split('A')
                filenames.append(jpeg_file_path)
                if int(file_labels[1]) >= 0 and int(file_labels[1]) <= 3:
                    agelabels.append(0)
                elif int(file_labels[1]) >= 4 and int(file_labels[1]) <= 7:
                    agelabels.append(1)
                elif int(file_labels[1]) >= 8 and int(file_labels[1]) <= 14:
                    agelabels.append(2)
                elif int(file_labels[1]) >= 15 and int(file_labels[1]) <= 24:
                    agelabels.append(3)
                elif int(file_labels[1]) >= 25 and int(file_labels[1]) <= 37:
                    agelabels.append(4)
                elif int(file_labels[1]) >= 38 and int(file_labels[1]) <= 47:
                    agelabels.append(5)
                elif int(file_labels[1]) >= 48 and int(file_labels[1]) <= 59:
                    agelabels.append(6)
                elif int(file_labels[1]) >= 60 and int(file_labels[1]) <= 100:
                    agelabels.append(7)
                
        
        shuffled_index = list(range(len(filenames)))
        random.seed(4321)
        random.shuffle(shuffled_index)
        
        filenames = [filenames[i] for i in shuffled_index]
        agelabels = [agelabels[i] for i in shuffled_index]
        genderlabels = [genderlabels[i] for i in shuffled_index]

        if istrain:
            self.train_filenames = filenames
            self.train_agelabels = agelabels
            self.train_genderlabels = genderlabels
        else:
            self.valid_filenames = filenames
            self.val_agelabels = agelabels
            self.val_genderlabels = genderlabels

    def process_image_files(self, name, filenames, labels, num_shards):
        assert len(filenames) == len(labels)

        # Break all images into batches with a [ranges[i][0], ranges[i][1]].
        spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
        ranges = []
        threads = []
        for i in range(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i+1]])

        # Launch a thread for each batch.
        print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
        sys.stdout.flush()

        # Create a mechanism for monitoring when all threads are finished.
        coord = tf.train.Coordinator()

        threads = []
        for thread_index in range(len(ranges)):
            args = (thread_index, ranges, name, filenames, labels, num_shards)
            t = threading.Thread(target=self.process_image_batch, args=args)
            t.start()
            threads.append(t)

        # Wait for all the threads to terminate.
        coord.join(threads)
        print('%s: Finished writing all %d images in data set.' % (datetime.now(), len(filenames)))
        sys.stdout.flush()

    def process_image_batch(self, thread_index, ranges, name, filenames, labels, num_shards):
        num_threads = len(ranges)
        assert not num_shards % num_threads
        num_shards_per_batch = int(num_shards / num_threads)

        shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], num_shards_per_batch + 1).astype(int)
        num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

        counter = 0
        for s in range(num_shards_per_batch):
            # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
            shard = thread_index * num_shards_per_batch + s
            output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
            if len(set(labels)) == 2:
                output_file = os.path.join(FLAGS.output_gender_dir, output_filename)
            else:
                output_file = os.path.join(FLAGS.output_age_dir, output_filename)
            writer = tf.io.TFRecordWriter(output_file)
            
            shard_counter = 0
            files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
            for i in files_in_shard:
                filename = filenames[i]
                label = int(labels[i])

                image_buffer, height, width = self.process_an_image(filename)
                
                example = self.return_example(filename, image_buffer, label, height, width)
                writer.write(example.SerializeToString())
                shard_counter += 1
                counter += 1

                if not counter % 1000:
                    print('%s [thread %d]: Processed %d of %d images in thread batch.' % (datetime.now(), thread_index, counter, num_files_in_thread))
                    sys.stdout.flush()

            writer.close()
            print('%s [thread %d]: Wrote %d images to %s' % (datetime.now(), thread_index, shard_counter, output_file))
            sys.stdout.flush()
            shard_counter = 0
        print('%s [thread %d]: Wrote %d images to %d shards.' % (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()
    
    def process_an_image(self, filename):
    
        image_data = self.handler.get_image_data(filename)
        image = self.handler.resample_jpeg(image_data)

        return image, RESIZE_HEIGHT, RESIZE_WIDTH


def main(unused_argv):
    assert not FLAGS.train_shards % FLAGS.num_threads, ('Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.valid_shards % FLAGS.num_threads, ('Please make the FLAGS.num_threads commensurate with ''FLAGS.valid_shards')

    if os.path.exists(FLAGS.output_age_dir) is False:
        print('creating %s' % FLAGS.output_age_dir)
        os.makedirs(FLAGS.output_age_dir)
        
    if os.path.exists(FLAGS.output_gender_dir) is False:
        print('creating %s' % FLAGS.output_gender_dir)
        os.makedirs(FLAGS.output_gender_dir)

    processor = DataProcessor('%s/%s' % (FLAGS.list_dir, FLAGS.train_list), '%s/%s' % (FLAGS.list_dir, FLAGS.valid_list), FLAGS.data_dir, FLAGS.train_shards, FLAGS.valid_shards)
    processor.process_dataset()

    # age      
    output_file = os.path.join(FLAGS.output_age_dir, 'md.json')
    train, valid, train_outcomes = len(processor.train_agelabels), len(processor.val_agelabels), set(processor.train_agelabels)


    md = { 'num_valid_shards': FLAGS.valid_shards, 
           'num_train_shards': FLAGS.train_shards,
           'valid_counts': valid,
           'train_counts': train,
           'timestamp': str(datetime.now()),
           'nlabels': len(train_outcomes) }
    with open(output_file, 'w') as f:
        json.dump(md, f)

    # gender  
    output_file = os.path.join(FLAGS.output_gender_dir, 'md.json')
    train, valid, train_outcomes = len(processor.train_genderlabels), len(processor.val_genderlabels), set(processor.train_genderlabels)

    md = { 'num_valid_shards': FLAGS.valid_shards, 
           'num_train_shards': FLAGS.train_shards,
           'valid_counts': valid,
           'train_counts': train,
           'timestamp': str(datetime.now()),
           'nlabels': len(train_outcomes) }
    with open(output_file, 'w') as f:
        json.dump(md, f)

if __name__ == '__main__':
    tf.compat.v1.app.run()