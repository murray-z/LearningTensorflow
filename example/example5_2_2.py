# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np


# Padding
seqlens = []
data = []
with open('../example5_data/data.txt', 'r') as f:
    for line in f:
        lis = line.strip().split()
        len_line = len(lis)
        seqlens.append(len_line)
        if len_line < 6:
            lis.extend(['PAD'] * (6 - len_line))
            data.append(' '.join(lis))

# Map
word2index_map = {}
index = 0
for sent in data:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

index2word_map = {index: word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map)

# Label ==> one_hot
labels = [1]*10000 + [0]*10000
for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0]*2
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding

# get train_data test_data
data_indices = list(range(len(data)))
np.random.shuffle(data_indices)

data = np.array(data)[data_indices]
labels = np.array(labels)[data_indices]
seqlens = np.array(seqlens)[data_indices]

train_x = data[:10000]
train_y = labels[:10000]
train_seqlens = seqlens[:10000]

test_x = data[10000:]
test_y = labels[10000:]
test_seqlens = seqlens[10000:]


def get_sentence_batch(batch_size, data_x, data_y, data_seqlens):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word2index_map[word] for word in data_x[i].lower().split()] for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x, y, seqlens


batch_size = 128
embedding_dimension = 64
num_classes = 2
hidden_layer_size = 32
times_steps = 6
element_size = 1

# Placeholder
_inputs = tf.placeholder(tf.int32, shape=[batch_size, times_steps])
_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
_seqlens = tf.placeholder(tf.int32, shape=[batch_size])


# Embedding
with tf.name_scope('embeddings'):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dimension], -1, 1), name='embedding')
    embed = tf.nn.embedding_lookup(embeddings, _inputs)

# Lstm
with tf.variable_scope('lstm'):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, embed, sequence_length=_seqlens, dtype=tf.float32)

# weight
weights = {'linear_layer': tf.Variable(tf.truncated_normal([hidden_layer_size,
                                                            num_classes], mean=0, stddev=0.01))}

# bias
biases = {'linear_layer': tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=0.01))}

# loss
final_output = tf.matmul(states[1], weights['linear_layer']) + biases['linear_layer']
softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=_labels)
cross_entropy = tf.reduce_mean(softmax)
tf.summary.scalar('loss', cross_entropy)

# acc
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
correct_predicton = tf.equal(tf.argmax(_labels, 1), tf.argmax(final_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predicton, tf.float32)) * 100
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

# train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_summary = tf.summary.FileWriter('./log_lstm/train', graph=tf.get_default_graph())
    test_summary = tf.summary.FileWriter('./log_lstm/test', graph=tf.get_default_graph())

    for step in range(1000):
        x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size, train_x, train_y, train_seqlens)
        summary, _ = sess.run([merged, train_step], feed_dict={_inputs: x_batch, _labels: y_batch, _seqlens: seqlen_batch})
        train_summary.add_summary(summary, step)

        x_batch_test, y_batch_test, seqlen_batch_test = get_sentence_batch(batch_size, test_x, test_y, test_seqlens)
        summary = sess.run(merged, feed_dict={_inputs: x_batch_test, _labels: y_batch_test, _seqlens: seqlen_batch_test})
        test_summary.add_summary(summary, step)

        if step % 100 == 0:
            acc_train = sess.run(accuracy, feed_dict={_inputs: x_batch, _labels: y_batch, _seqlens: seqlen_batch})
            acc_test = sess.run(accuracy, feed_dict={_inputs: x_batch_test, _labels: y_batch_test, _seqlens: seqlen_batch_test})
            print('step: {}, acc_train:{:.5f}, acc_test:{:.5f}'.format(step, acc_train, acc_test))






