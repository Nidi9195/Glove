import zipfile
import collections
from math import log
import random
import numpy as np
import itertools
from scipy import sparse

''' Global variable declaration '''

f1 = "text8.zip"
window_size = 3
size = 253855 #no of most common words
#size = 259
epoch = 4
lr = 0.1
word_index_dict = {}
word_co_dict = {}

''' Global variable declaration ends '''

def read_data(filename):
    ''' Returns list of all words from zipfile '''
    with zipfile.ZipFile(filename) as f:
        m = f.read(f.namelist()[0]).decode('UTF-8')
        n = m.split()
        return n

def build_vocab(vocab_list):
    global size, word_index_dict
    count = []
    count.extend(collections.Counter(vocab_list).most_common(size - 1))
    word_index_dict['UNK']=0
    for var in range(0,size-1):
        word_index_dict[count[var][0]] = var+1
    print("Built dictionary!")

def build_cooccurrence_matrix(vocab):
    global size, word_index_dict, word_co
    vocab_size = len(vocab)
    for i in range(size - window_size):
        for j in range(1,window_size):
            try:
                word_co_dict[(vocab[i],vocab[i+j])]+=1
            except:
                try:
                    word_co_dict[(vocab[i+j],vocab[i])]+=1
                except:
                    word_co_dict[(vocab[i],vocab[i+j])] = 1
    #print(word_co_dict)
    #cooccurrences = sparse.lil_matrix((vocab_size, vocab_size))
    
    return 3

def main():
    global word_index_dict
    vocabulary = read_data(f1)
    print(len(vocabulary))
    print(len(set(vocabulary)))
    print("read text8")
    build_vocab(vocabulary)
    print(build_cooccurrence_matrix(vocabulary))

if __name__== "__main__":
    main()
