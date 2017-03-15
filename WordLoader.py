import tensorflow as tf
import tflearn
import logging

import numpy as np
from numpy import dtype, fromstring, float32 as REAL

class WordLoader(object):
    def load_word_vector(self, fname, binary=None):
        if binary == None:
            if fname.endswith('.txt'):
                binary = False
            elif fname.endswith('.bin'):
                binary = True
            else:
                raise NotImplementedError('Cannot infer binary from %s' % (fname))

        vocab = {}
        with open(fname) as fin:
            header = fin.readline()
            vocab_size, vec_size = map(int, header.split())  
            if binary:
                binary_len = dtype(REAL).itemsize * vec_size
                for line_no in xrange(vocab_size):
                    try:
                        word = []
                        while True:
                            ch = fin.read(1)
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        vocab[word] = fromstring(fin.read(binary_len), dtype=REAL)
                    except:
                        pass
            else:
                for line_no, line in enumerate(fin):
                    try:
                        parts = line.strip().split(' ')
                        word, weights = parts[0], map(REAL, parts[1:])
                        vocab[word] = weights
                    except:
                        pass
        return vocab

	def genWordVec(self, words, dim_r):
		num = len(words)
		logging.info('vocabulary size: %s' % num)
		
		fname = 'dataset/' + word_vector
		
		logging.info('loading word vectors...')
		dic = self.load_word_vector(fname)

		value = np.array((num + 1, dim_r), dtype=tf.float32)
		not_found = 0
		
		for words, index in words.items():
			word_list = eval(words)
			if word_list in dict:
				value[index] = list(dic[word_list[0]])
			else:
				not_found += 1
			
		logging.info('word vector for %s, %d not found.' % (key, not_found))
		logging.info('loading word vectors ok...')
		
		return value