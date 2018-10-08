# -*- coding: utf-8 -*-

"""
构造虚拟文本数据集
"""

import numpy as np

digit_to_word_map = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                     6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}


digit_to_word_map[0] = 'PAD'

even_sentences = []
odd_sentences = []

seqlens = []

for i in range(10000):
    rand_seq_len = np.random.choice(range(3, 7))
    seqlens.append(rand_seq_len)
    rand_odd_ints = np.random.choice(range(1, 10, 2), rand_seq_len)
    rand_even_ints = np.random.choice(range(2, 10, 2), rand_seq_len)

    # padding
    # if rand_seq_len < 6:
    #     rand_odd_ints = np.append(rand_odd_ints, [0]*(6-rand_seq_len))
    #     rand_even_ints = np.append(rand_even_ints, [0]*(6-rand_seq_len))

    even_sentences.append(' '.join([digit_to_word_map[i] for i in rand_even_ints]))
    odd_sentences.append(' '.join([digit_to_word_map[i] for i in rand_odd_ints]))

data = even_sentences + odd_sentences

with open('../example5_data/data.txt', 'w') as f:
    f.write('\n'.join(data))