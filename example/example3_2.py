# -*- coding: utf-8 -*-

"""
@Time    : 2018/10/1 14:26
@Author  : fazhanzhang
@Function :
"""

import numpy as np
import tensorflow as tf


N = 20000
NUM_STEPS = 50


def sigmoid(x):
    return 1/(1+np.exp(-x))


x_data = np.random.randn(N, 3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2

wxb = np.matmul(w_real, x_data.T) + b_real

y_data_pre_noise = sigmoid(wxb)

with tf.name_scope('placeholder'):
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_true = tf.placeholder(tf.float32, shape=None)

with tf.name_scope('inference'):
    w = tf.Variable([[0, 0, 0]], dtype=tf.float32, name='weight')
    # w = tf.Variable(tf.random_normal([1, 3]), dtype=tf.float64, name='weight')
    b = tf.Variable(0, dtype=tf.float32, name='bias')
    y_pred = tf.matmul(w, tf.transpose(x)) + b

with tf.name_scope('loss'):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    loss = tf.reduce_mean(loss)

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(NUM_STEPS):
        sess.run(train, feed_dict={x: x_data, y_true: y_data_pre_noise})
        if step % 5 == 0:
            print(step, sess.run([w, b]))
    print(50, sess.run([w, b]))
