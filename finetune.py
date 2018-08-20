#############################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-07-31
#############################################

"""This is the implementation of fine-tuning a model"""

from nn_ops import dense, flatten
import tensorflow as tf
from collections import OrderedDict
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as model
from utils import load_data_sets, l2_loss, train


def resnet50v1(inputs, output_dims, training=True):

    # transfer learning with pre-trained model
    net, _ = model.resnet_v1.resnet_v1_50(inputs, output_dims, is_training=training)

    graph = tf.get_default_graph()

    avg_pool = graph.get_tensor_by_name('resnet_v1_50/pool5:0')
    avg_pool = flatten(avg_pool)

    with tf.variable_scope('resnet_v1_50/fc6'):
        fc6 = dense(avg_pool, output_dims, activation=None)

    return fc6


if __name__ == '__main__':

    image_size  = [450, 450, 3]
    output_dims = 398
    model_path  = 'pre-trained-models/resnet_v1_50.ckpt'
    train_list  = "list/train.list"
    val_list    = "list/dev.list"
    logdir      = "train-logs"
    num_epochs  = 50
    batch_size  = 32

    inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='inputs')
    truths = tf.placeholder(tf.float32, [None, output_dims], name='truths')
    training = tf.placeholder(tf.bool, name='training')

    predicts = resnet50v1(inputs, output_dims, training)
    with tf.name_scope('object_loss'):
        obj_loss = l2_loss(predicts, truths)
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = obj_loss + tf.add_n(reg_loss)

    tf.summary.scalar('total loss', loss)
    tf.summary.scalar('obj loss', obj_loss)

    lr_rate = 0.001
    var_list = [v for v in tf.trainable_variables()]
    with tf.name_scope('train'):
        optm = tf.train.MomentumOptimizer(learning_rate=lr_rate, momentum=0.9)
        grads_vars = optm.compute_gradients(loss, var_list)
        train_op = optm.apply_gradients(grads_and_vars=grads_vars)

    exclude = ['resnet_v1_50/logits/weights', 'resnet_v1_50/logits/biases',
               'resnet_v1_50/fc6/weights', 'resnet_v1_50/fc6/biases']
    vars_restore = slim.get_variables_to_restore(exclude=exclude)

    restorer = tf.train.Saver(var_list=vars_restore)
    saver = tf.train.Saver(max_to_keep=10)

    feed_dict = OrderedDict.fromkeys([inputs, truths, training])
    group_op = tf.group(train_op)

    data_family = load_data_sets(image_size[0], output_dims, batch_size, train_list, val_list)

    # begin training
    train(group_op, loss, feed_dict, data_family, num_epochs, saver, restorer, model_path)




