#############################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-07-31
#############################################

"""This is implementation of batch preprocess."""

import os
import glob
import numpy as np
import collections
import tensorflow as tf
# from os import listdir
# from os.path import isfile, join
from datetime import datetime
from tensorflow.data import Dataset
from tensorflow.data import Iterator
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

TRAIN_SET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

DataMember = collections.namedtuple("DataMember", ['init_op', 'num_examples', 'batch_size'])

DataFamily = collections.namedtuple("DataFamily", ['train', 'val', 'next_batch'])


class DataGenerator(object):

    def __init__(self, base_dir, txt_file, output_dim, mode, batch_size, shuffle=True,
                 buffer_size=10000):

        self.base_dir = base_dir
        self.txt_file = os.path.join(base_dir, txt_file)
        self.output_dim = output_dim

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.float32)

        # slice first dimension of tensor to create dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls=8)
        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_parallel_calls=8)
        else:
            raise ValueError("Invalid mode '%s'." % mode)

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _read_txt_file(self):
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip()  # strip the last '\n'
                item_path = os.path.join(self.base_dir, item)
                self.img_paths.append(item_path + 'mix.png')
                # read y data, stored in .txt
                param_list = read_txt(item_path + '.shape')
                color_list = read_txt(item_path + '.color')
                exp_list = read_txt(item_path + '.expression')
                param_list.extend(color_list)
                param_list.extend(exp_list)
                self.labels.append(param_list)

    def _shuffle_lists(self):
        img_paths = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(img_paths[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, img_file, label):

        img_normed = self.parse_data(img_file)

        label_normed = label

        return img_normed, label_normed

    def _parse_function_inference(self, img_file, label):

        img_normed = self.parse_data(img_file)

        label_normed = label

        return img_normed, label_normed

    @staticmethod
    def parse_data(img_file):
        # load and pre-process the image
        img_string = tf.read_file(img_file)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        # img_resized = tf.image.resize_images(img_decoded, [224, 224])
        img_normed = tf.divide(tf.cast(img_decoded, tf.float32), 255.)

        return img_normed


def read_txt(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    num_list = []
    for line in lines:
        tmp = list(map(float, line.strip('\n').strip().split(' ')))
        num_list += tmp
    return num_list


def load_data_sets(output_dim, batch_size, train_list, val_list, base_dir):

    # Place data loading and pre-processing on cpu
    with tf.device('/cpu:0'):
        train_data = DataGenerator(base_dir, train_list, output_dim, 'training', batch_size, shuffle=True)
        val_data = DataGenerator(base_dir, val_list, output_dim, 'inference', batch_size, shuffle=False)

    # Create an reinitializable iterator given the data structure
    iterator = Iterator.from_structure(train_data.data.output_types, train_data.data.output_shapes)
    next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    train_init_op = iterator.make_initializer(train_data.data)
    val_init_op = iterator.make_initializer(val_data.data)

    train_member = DataMember(init_op=train_init_op, num_examples=train_data.data_size, batch_size=batch_size)
    val_member = DataMember(init_op=val_init_op, num_examples=val_data.data_size, batch_size=batch_size)

    return DataFamily(train=train_member, val=val_member, next_batch=next_batch)


def load_data_set():
    pass


def train(group_op, loss, feeds, dataset, epochs, saver, logdir, restorer, model_path=None, show_step=100):

    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(logdir + '/train')
    dev_writer = tf.summary.FileWriter(logdir + '/dev')

    train_batches = int(np.ceil(dataset.train.num_examples / dataset.train.batch_size))
    val_batches = int(np.ceil(dataset.val.num_examples / dataset.val.batch_size))

    train_init_op = dataset.train.init_op
    val_init_op = dataset.val.init_op

    next_batch = dataset.next_batch

    # # begin training
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        if model_path:
            restorer.restore(sess, model_path)

        sess.run(tf.global_variables_initializer())

        train_writer.add_graph(sess.graph)

        print("{} begin training...".format(datetime.now().strftime(TIME_FORMAT)))

        for epoch in range(epochs):

            print("{} epoch number: {}".format(datetime.now().strftime(TIME_FORMAT), epoch + 1))

            # Training here
            sess.run(train_init_op)

            for batch in range(train_batches):

                x_batch, y_batch = sess.run(next_batch)

                feed_dict = dict(zip([*feeds], [x_batch, y_batch, True]))
                _, _, obj_loss = sess.run([group_op, loss], feed_dict=feed_dict)

                rel_loss = obj_loss / np.mean(np.square(y_batch))

                if batch % show_step == 0:
                    print("{}, epoch: {}, loss: {:.8f}, rel loss: {:.9f}"
                          .format(datetime.now().strftime(TIME_FORMAT), epoch + 1, obj_loss, rel_loss))

                    feed_dict = dict(zip([*feeds], [x_batch, y_batch, False]))
                    s = sess.run(merged_summary, feed_dict=feed_dict)
                    train_writer.add_summary(s, epoch * train_batches + batch)

            # Validating here
            print("{} begin validation".format(datetime.now().strftime(TIME_FORMAT)))

            sess.run(val_init_op)

            loss_list = []

            for b in range(val_batches):

                x_batch, y_batch = sess.run(next_batch)

                feed_dict = dict(zip([*feeds], [x_batch, y_batch, False]))
                obj_loss = sess.run([loss], feed_dict=feed_dict)

                loss_list.append(obj_loss)

                if b % show_step == 0:
                    s = sess.run(merged_summary, feed_dict=feed_dict)
                    dev_writer.add_summary(s, epoch * train_batches + b)

            average_loss = np.mean(loss_list)
            print("{}, epoch: {}, val loss: {:.8f}".format(datetime.now().strftime(TIME_FORMAT), epoch + 1, average_loss))

            # # checkpoint save
            checkpoint_name = os.path.join(logdir + '/ckpts', 'model_epoch' + str(epoch + 1) + '.ckpt')
            saver.save(sess, checkpoint_name)
            print("{} model checkpoint saved at {}".format(datetime.now().strftime(TIME_FORMAT), checkpoint_name))


def get_file_list(files_path, save_file):
    # 1st method
    file_list = glob.glob(os.path.join(files_path, '*.png'))

    # 2nd method
    # only_files = [f for f in listdir(files_path) if isfile(join(files_path, f))]

    with open(save_file, 'w') as f:
        f.write('\n'.join(file_list))  # write(a single string) while writelines(list of string)
