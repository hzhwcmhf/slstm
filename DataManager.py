#coding:utf-8

import numpy as np
import theano
import codecs
import random
random.seed(1229)

class DataManager(object):
	def __init__(self, data):
		def load_data(fname):
			data = []
			with open(fname) as f:
				for line in f:
					now = line.strip().split()
					data.append((int(now[0]), now[1:]))
			return data

		self.origin_data = {}
		for fname in ['train', 'dev', 'test']:
			self.origin_data[fname] = load_data('dataset/%s/%s.txt' % (data, fname))
		self.origin_words = {}
			

	def gen_word_list(self):
		words = {}
		for key in ['train', 'dev', 'test']:
			for label, sent in self.origin_data[key]:
				for word in sent:
					if repr([word]) not in words.keys():
						id = len(words)
						words[repr([word])] = id
		self.words = words
		return self.words
	
	def gen_data(self):
		
		self.grained = 1 + max([x for data in self.origin_data.values() for x, y in data])

		self.data = {}
		for key in ['train', 'dev', 'test']:
			data = []
			label = []
			index = 0
			for rating, sent in self.origin_data[key]:
				# print single
				result = [self.words(repr[x]) for x in sent]
					
				data.append(np.array(result))
				rat = np.zeros((self.grained), dtype = theano.config.floatX)
				rat[rating] = 1
				label.append(rat)
				index = index + 1
			self.data[key] = (label, data)
		
		self.data['train_small'] = self.data['train'][0][::10], self.data['train'][1][::10], 
		
		self.index = list(range(len(self.data['train'][0])))
		self.index_now = 0
		return self.data

	def get_mini_batch(self, mini_batch_size=25):
		if self.index_now >= len(self.index):
			random.shuffle(self.index)
			self.index_now = 0
		st, ed = self.index_now, self.index_now + mini_batch_size
		label = np.take(self.data['train'][0], self.index[st:ed], 0)
		data = np.take(self.data['train'][1], self.index[st:ed], 0)
		self.index_now += mini_batch_size
		return label, data
