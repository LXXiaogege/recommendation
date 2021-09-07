import pickle

import bcolz
import numpy as np
import torch

glove_path = "D:\data\glove.6B"


def parse_file(glove_path):
    words = []
    idx = 0
    word2idx = {}

    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.200.dat', mode='w')

    with open(f'{glove_path}/glove.6B.200d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400000, 200)), rootdir=f'{glove_path}/6B.200.dat', mode='w')
    vectors.flush()
    torch.save(words, open(f'{glove_path}/6B.200_words.pkl', 'wb'))
    torch.save(word2idx, open(f'{glove_path}/6B.200_idx.pkl', 'wb'))


"""
获取词向量
"""
vectors = bcolz.open(f'{glove_path}/6B.200.dat')[:]
words = torch.load(open(f'{glove_path}/6B.200_words.pkl', 'rb'))
word2idx = torch.load(open(f'{glove_path}/6B.200_idx.pkl', 'rb'))
glove = {w: torch.from_numpy(vectors[word2idx[w]]) for w in words}
print(glove['the'])

