import numpy as np
import tensorflow as tf
from nn_ops import dense, flatten
import tensorflow.contrib.slim.nets as model


def resnet50v1(inputs, output_dims, training=True):

    # transfer learning with pre-trained model
    net, _ = model.resnet_v1.resnet_v1_50(inputs, output_dims, is_training=training)

    graph = tf.get_default_graph()

    avg_pool = graph.get_tensor_by_name('resnet_v1_50/pool5:0')
    avg_pool = flatten(avg_pool)

    with tf.variable_scope('resnet_v1_50/fc6'):
        fc6 = dense(avg_pool, output_dims, activation=None)

    return fc6


shape_weights_loss = np.genfromtxt("norm_variance.shape") * 199  # np.ones(199)
color_weights_loss = np.genfromtxt("norm_variance.color") * 199  # np.ones(199)
exp_weights_loss = np.genfromtxt('norm_variance.expression') * 100


def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))


def mae_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred - y_true))


def wgt_loss_shape(y_true, y_pred):
    shape_true = y_true[:, :199]
    shape_pred = y_pred[:, :199]
    return tf.reduce_mean(tf.square(shape_true - shape_pred) * shape_weights_loss)


def wgt_loss_color(y_true, y_pred):
    color_true = y_true[:, 199:398]
    color_pred = y_pred[:, 199:398]
    return tf.reduce_mean(tf.square(color_true - color_pred) * color_weights_loss)


def wgt_loss_exp(y_true, y_pred):
    exp_true = y_true[:, 398:]
    exp_pred = y_pred[:, 398:]
    return tf.reduce_mean(tf.square(exp_true - exp_pred) * exp_weights_loss)


def dist_loss(y_true, y_pred):
    shape_true = y_true[:, :199]
    shape_pred = y_pred[:, :199]

    color_true = y_true[:, 199:398]
    color_pred = y_pred[:, 199:398]

    exp_true = y_true[:, 398:]
    exp_pred = y_pred[:, 398:]

    shape_mean, shape_var = tf.nn.moments(shape_pred, axes=[1])
    shape_loss_mean_dist = tf.reduce_mean(tf.square(shape_mean))
    shape_loss_variance_dist = tf.reduce_mean(tf.square(shape_var - 1))

    color_mean, color_var = tf.nn.moments(color_pred, axes=[1])
    color_loss_mean_dist = tf.reduce_mean(tf.square(color_mean))
    color_loss_variance_dist = tf.reduce_mean(tf.square(color_var - 1))

    exp_mean, exp_var = tf.nn.moments(exp_pred, axes=[1])
    exp_loss_mean_dist = tf.reduce_mean(tf.square(exp_mean))
    exp_loss_variance_dist = tf.reduce_mean(tf.square(exp_var - 1))

    return shape_loss_mean_dist + shape_loss_variance_dist + color_loss_mean_dist + color_loss_variance_dist \
        + exp_loss_mean_dist + exp_loss_variance_dist


def my_custom_loss(y_true, y_pred):
    return mse_loss(y_true, y_pred) + wgt_loss_shape(y_true, y_pred) + wgt_loss_color(y_true, y_pred) \
        + wgt_loss_exp(y_true, y_pred) + 0.2 * dist_loss(y_true, y_pred)
