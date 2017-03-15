# coding:utf-8

import tensorflow as tf
import tflearn
from tflearn import variables as va
from tflearn import initializations
from tflearn import activations

def PhraseLayer(incoming, input_dim, output_dim, output_length, activation='linear', 
		dropout_keepprob = 0.5, batchNorm = False, name = 'PhraseLayer', alpha = 0.5, scope = None):
	'''
	incoming: [batch_size, sen_length, input_dim]
	
	return: [batch_size, sen_length, output_length, output_dim[0]],
			[batch_size, sen_length, output_length, output_dim[1]]
	'''
	with tf.variable_scope(scope, default_name=name, values=[incoming]) as scope:
		name = scope.name
		
		P = va.variable('P', shape=[input_dim, output_dim[0]],
				initializer=initializations.get('truncated_normal')())
		tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, P)
		
		P_p = va.variable('P_p', shape=[input_dim, output_dim[1]],
				initializer=initializations.get('truncated_normal')())
		tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, P_p)
		
		Q = va.variable('Q', shape=[input_dim, output_dim[0]],
				initializer=initializations.get('truncated_normal')())
		tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, Q)
		
		Q_p = va.variable('Q_p', shape=[input_dim, output_dim[1]],
				initializer=initializations.get('truncated_normal')())
		tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, Q_p)
		
		R = va.variable('R', shape=[input_dim, output_dim[0]],
				initializer=initializations.get('truncated_normal')())
		tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, R)
		
		R_p = va.variable('R_p', shape=[input_dim, output_dim[1]],
				initializer=initializations.get('truncated_normal')())
		tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, R_p)
		
		O = va.variable('O', shape=[output_dim[0], output_dim[0]],
				initializer=initializations.get('truncated_normal')())
		tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, O)
		
		O_p = va.variable('O_p', shape=[output_dim[1], output_dim[1]],
				initializer=initializations.get('truncated_normal')())
		tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, O_p)
		
		b = va.variable('b', shape=[1, 1, 1, output_dim[0]],
				initializer=initializations.get('zeros')())
		
		b_p = va.variable('b_p', shape=[1, 1, 1, output_dim[1]],
				initializer=initializations.get('zeros')())
				
		
		if isinstance(activation, str):
			activation = activations.get(activation)
		elif hasattr(activation, '__call__'):
			activation = activation
		else:
			raise ValueError("Invalid Activation.")
		
		
		def calc(incoming, P, Q, R, O, output_dim):
		
			batch_size = tf.shape(incoming)[0]
			sent_length = tf.shape(incoming)[1]
			
			G1 = tf.zeros([batch_size * sent_length, output_dim])
			G2 = tf.zeros([batch_size * sent_length, output_dim])
			G3 = tf.zeros([batch_size * sent_length, output_dim])
			r = []
			
			for i in range(output_length):
				
				if i == 0:
					now = incoming
				else:
					now = tf.concat([tf.zeros([batch_size, i, input_dim]), incoming[:,0:-i, :]], axis = 1)
				
				now = tf.reshape(incoming, [batch_size * sent_length, input_dim])
				
				F1 = tf.matmul(now, P) if i == 0 else tf.zeros([batch_size * sent_length, output_dim])
				F2 = tf.matmul(now, Q) * G1
				F3 = tf.matmul(now, R) * G2
				
				G1 = G1 * alpha + F1
				G2 = G2 * alpha + F2
				G3 = G3 * alpha + F3
				
				r.append(tf.matmul(G1+G2+G3, O))
				
			return tf.reshape(tf.stack(r, axis = 1), [batch_size, sent_length, output_length, output_dim])
		
		out1 = calc(incoming, P, Q, R, O, output_dim[0]) + b
		out2 = calc(tf.stop_gradient(incoming), P_p, Q_p, R_p, O_p, output_dim[1]) + b_p
		
		out1 = activation(out1, name="activation")
		out2 = activation(out2, name="activation_p")
		
		tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, out1)
		tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, out2)
		
		if batchNorm:
			out1 = tflearn.batch_normalization(out1, name="batchNormOut1")
			out2 = tflearn.batch_normalization(out2, name="batchNormOut2")
		
		out1 = tflearn.dropout(out1, dropout_keepprob, name="dropOut1")
		out2 = tflearn.dropout(out2, dropout_keepprob, name="dropOut2")
	
	out1.seq_length = incoming.seq_length
	out2.seq_length = incoming.seq_length
	
	tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, out1)
	tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, out2)
		
	return out1, out2
	