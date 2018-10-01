# -*- coding: utf-8 -*-

"""
@Time    : 2018/10/1 10:52
@Author  : fazhanzhang
@Function : 线性回归
"""

import numpy as np
import tensorflow as tf


# 构造数据
x_data = np.random.randn(2000, 3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2

noise = np.random.randn(1, 2000) * 0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise

# 模型及训练
NUM_STEP = 10
wb_ = []

x = tf.placeholder(tf.float32, shape=[None, 3])
y_true = tf.placeholder(tf.float32, shape=None)

with tf.name_scope('inference') as scope:
    # w = tf.Variable([[0, 0, 0]], dtype=tf.float32, name='weight')
    w = tf.Variable(tf.random_normal([1, 3]), dtype=tf.float32, name='weight')
    b = tf.Variable(0, dtype=tf.float32, name='bias')
    y_pred = tf.matmul(w, tf.transpose(x)) + b

with tf.name_scope('loss') as scope:
    loss = tf.reduce_mean(tf.square(y_true - y_pred))

with tf.name_scope('train') as scope:
    learning_rate = 0.5
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(NUM_STEP):
        sess.run(train, feed_dict={x: x_data, y_true: y_data})
        if step % 5 == 0:
            print(step, sess.run([w, b]))
            wb_.append(sess.run([w, b]))

    print(10, sess.run([w, b]))