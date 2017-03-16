# coding:utf-8

import tensorflow as tf
import tflearn
from tflearn import variables as va

def ClassifyLayer(incoming, dim, activation = 'prelu', keepdrop = 0.8, scope = None, name = "ClassifyLayer"):
	
	'''
	incomming:
		[batch_size, feature]
	output:
		[batch_size, dim[-1]]
	'''
	
	with tf.variable_scope(scope, default_name=name, values=[incoming]) as scope:
		inference = incoming
		for d in dim[:-1]:
			inference = tflearn.fully_connected(inference, d, activation = activation, name="W"+str(d))
			inference = tflearn.dropout(inference, keepdrop)
		
		inference = tflearn.fully_connected(inference, dim[-1], activation = 'softmax', name="Wend")
	
	tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)
	
	return inference
	