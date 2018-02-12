import zipfile
from collections import Counter
import itertools
from functools import partial
from math import log
import operator
import random
import numpy as np
import pickle
from random import shuffle
from scipy import sparse
#from util import listify

''' Global variable declaration '''

f1 = "text_try.zip"
window_size = 3
epoch = 200
lr = 0.1
size = 50000 #no of most common words
inputlayer_neurons = size
hiddenlayer_neurons = 300
output_neurons = size

def read_data(filename):
    ''' Returns list of all words from zipfile '''
    with zipfile.ZipFile(filename) as f:
        m = f.read(f.namelist()[0]).decode('UTF-8')
        n = m.split()
        return m,n

def build_vocab(corpus):
    print("Building vocab from corpus")
    vocab = Counter()
    for line in corpus:
        tokens = line.strip().split()
        vocab.update(tokens)
    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.items())}

def build_cooccur(vocab, corpus):
    print("Building cooccur")
    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.items())
    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),dtype=np.float64)
    for i, line in enumerate(corpus):
        if i % 1000 == 0:
            print("Building cooccurence matrix for line ", i)
        tokens = line.strip().split()
        tokens_ids = [vocab[word][0] for word in tokens]

        for center_i, center_id in enumerate(token_ids):
            context_ids = token_ids[max(0, center_i - window_size) : center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                distance = contexts_len - left_i

                increment = 1.0 / float(distance)

                cooccurrences[center_id, left_id] += increment
                cooccurrences[center_id, left_id] += increment

    for i, (row, data) in enumerate(itertools.izip(cooccurrences.rows,cooccurrences.data)):
        if min_count is not None and vocab[id2word[i]][0] < min_count:
            continue

        for data_idx, j in enumerate(row):
            if min_count is not None and vocab[id2word[j]][0] < min_count:
                continue
            yield i, j, data[data_idx]


def main():
    vocabulary_str, vocabulary = read_data(f1)
    vocab = build_vocab(vocabulary_str)
    cooccurrences = build_cooccur(vocab, vocabulary_str)


if __name__== "__main__":
    main()
