# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../mnist', one_hot=True)

element_size = 28
time_step = 28
num_class = 10
batch_size = 128
hidden_layer_size = 128

_inputs = tf.placeholder(tf.float32, shape=[None, time_step, element_size], name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_class], name='inputs')

# rnn
rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
outputs, _ = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)

Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_class],
                                     mean=0, stddev=0.01))
bl = tf.Variable(tf.truncated_normal([num_class], mean=0, stddev=0.01))


def get_linear_layer(vertor):
    return tf.matmul(vertor, Wl) + bl


last_rnn_output = outputs[:, -1, :]
final_output = get_linear_layer(last_rnn_output)
softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=y)
cross_entropy = tf.reduce_mean(softmax)
tf.summary.scalar('loss', cross_entropy)

train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(final_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))*100
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_summary = tf.summary.FileWriter('./log_rnn_mnist/train', graph=tf.get_default_graph())
    test_summary = tf.summary.FileWriter('./log_rnn_mnist/test', graph=tf.get_default_graph())

    test_data = mnist.test.images.reshape((-1, time_step, element_size))
    test_label = mnist.test.labels

    for i in range(3001):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, time_step, element_size))
        summary_train, _ = sess.run([merged, train_step], feed_dict={_inputs: batch_x, y: batch_y})

        train_summary.add_summary(summary_train, i)

        summary_test, _ = sess.run([merged, train_step], feed_dict={_inputs: test_data, y: test_label})

        train_summary.add_summary(summary_test, i)

        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict={_inputs: batch_x, y: batch_y})
            loss = sess.run(cross_entropy, feed_dict={_inputs: batch_x, y: batch_y})
            print("Iter: {}, loss: {:.6f}, acc: {:.5f}".format(i, loss, acc))
            print("Test acc: {}".format(sess.run(accuracy, feed_dict={_inputs: test_data, y: test_label})))
