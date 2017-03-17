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
			sent_length = incoming.shape[1].value
			
			G1 = tf.zeros([batch_size, sent_length, output_dim])
			G2 = tf.zeros([batch_size, sent_length, output_dim])
			G3 = tf.zeros([batch_size, sent_length, output_dim])
			r = []
			
			for i in range(output_length):
				
				if i == 0:
					now = incoming
				else:
					now = tf.concat([tf.zeros([batch_size, i, input_dim]), incoming[:,0:-i, :]], axis = 1)
				
				F2 = tf.einsum('aij,jk->aik', now, Q) * G1
				F3 = tf.einsum('aij,jk->aik', now, R) * G2
				
				if i == 0:
					F1 = tf.einsum('aij,jk->aik', now, P)
					G1 = G1 * alpha + F1
				else:
					G1 = G1 * alpha
				G2 = G2 * alpha + F2
				G3 = G3 * alpha + F3
				
				r.append(tf.einsum('aij,jk->aik',G1+G2+G3, O))
				
			return tf.stack(r, axis = 2)
		
		#batch_size = tf.shape(incoming)[0]
		#sent_length = incoming.shape[1].value
		#out1 = tf.reshape(tf.einsum('aij,jk->aik', incoming, P), [batch_size, sent_length, 1, output_dim[0]])
		out1 = calc(incoming, P, Q, R, O, output_dim[0]) + b
		#out1 = activation(out1, name="activation")
		tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, out1)
		if batchNorm:
			pass
			#out1 = tflearn.batch_normalization(out1, name="batchNormOut1")
		#out1 = tflearn.dropout(out1, dropout_keepprob, name="dropOut1")
		
		if output_dim[1] == 0:
			out2 = None
		else:
			out2 = calc(tf.stop_gradient(incoming), P_p, Q_p, R_p, O_p, output_dim[1]) + b_p
			out2 = activation(out2, name="activation_p")
			tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, out2)
			if batchNorm:
				out1 = tflearn.batch_normalization(out1, name="batchNormOut1")
			out2 = tflearn.batch_normalization(out2, name="batchNormOut2")
			out2 = tflearn.dropout(out2, dropout_keepprob, name="dropOut2")
	
	tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, out1)
	if output_dim[1] != 0:
		tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, out2)
		
	return out1, out2
	