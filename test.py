import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST-data', one_hot=True)

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 784], name='input')

y = tf.layers.Dense(10)(x)
output = tf.nn.softmax(y, axis=1, name='output')

tf.contrib.quantize.create_eval_graph()

saver = tf.train.Saver()
saver.restore(sess, './local.ckpt')

with open('eval.pb', 'w') as f:
    g = tf.get_default_graph()
    f.write(str(g.as_graph_def()))

batch = mnist.train.next_batch(100)
results = sess.run(output, feed_dict={x: batch[0]})

truth = np.argmax(batch[1], -1)
predict = np.argmax(results, -1)
print(np.mean(truth == predict))
