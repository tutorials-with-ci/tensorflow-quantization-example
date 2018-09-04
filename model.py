import tensorflow as tf


def simple_conv_net(x, is_training: bool = False):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    x = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 2, 2)

    x = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 2, 2)

    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(x, 1024)
    x = tf.layers.dropout(x, rate=0.5, training=is_training)
    x = tf.layers.dense(x, 10)

    return x
