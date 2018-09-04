import tensorflow as tf
from model import simple_conv_net
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST-data', one_hot=True)
sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 784], name='input')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='label')

logits = simple_conv_net(x, is_training=True)
y = tf.nn.softmax(logits, name='prob')

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

tf.contrib.quantize.create_training_graph()

sess.run(tf.global_variables_initializer())
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(3000):
    batch = mnist.train.next_batch(128)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    if (i + 1) % 100 == 0:
        print('Iteration: {: 4d}'.format(i + 1))

saver = tf.train.Saver()
saver.save(sess, './local.ckpt')
