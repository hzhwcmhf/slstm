import tensorflow as tf

import numpy as np
import logging
from WordLoader import WordLoader


def WordVectorLayer(incoming, words, dim_word, word_vector, scope = None, reuse = False, name="WordVector"):
		""" 
			WordVectorLayer
			
		Input:
			2D Tensor [batch_size, length]
		Output:
			3D Tensor [batch_size, length, dim_word].
			
		"""
		
		num = len(words)
		logging.info('vocabulary size: %s' % num)
		
		fname = 'dataset/' + word_vector
		
		logging.info('loading word vectors...')
		loader = WordLoader()
		dic = loader.load_word_vector(fname)

		value = np.array(dim, dtype=tf.float32)
		not_found = 0
		
		for words, index in words.items():
			word_list = eval(words)
			if word_list in dict:
				value[index] = list(dic[word_list[0]])
			else:
				not_found += 1
			
		logging.info('word vector for %s, %d not found.' % (key, not_found))
		logging.info('loading word vectors ok...')
		
		with tf.variable_scope(scope, default_name=name, values=[incoming], reuse=reuse) as scope:
			name = scope.name
			
			with tf.device('/cpu:0'):
				V = va.variable('V', shape=[num, dim_word], initializer=value, trainable=True, restore=True)
				tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, V)
			
			inference = tf.cast(incoming, tf.int32)
			inference = tf.nn.embedding_lookup(V, inference, validate_indices=False)

	
		inference.W = W
		inference.scope = scope
		# Embedding doesn't support masking, so we save sequence length prior
		# to the lookup. Expand dim to 3d.
		shape = [-1] + inference.get_shape().as_list()[1:3] + [1]
		inference.seq_length = retrieve_seq_length_op(tf.reshape(incoming, shape))

		# Track output tensor.
		tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

		return inference



