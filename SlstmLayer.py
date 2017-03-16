# coding:utf-8

import tensorflow as tf
import tflearn
from tflearn import variables as va
from tflearn.layers.recurrent import BasicLSTMCell
from tensorflow.python.framework import ops

@ops.RegisterGradient("ST_OneHot")
def _ST_OneHot(op, grad):
	op.inputs[0].ST_grad = grad
	return [tf.zeros(tf.shape(op.inputs[0]), dtype=tf.int64), 
			tf.zeros([], dtype=tf.int32), 
			tf.zeros([], dtype=tf.float32), 
			tf.zeros([], dtype=tf.float32)]

@ops.RegisterGradient("ST_Multinomial")
def _ST_Multinomial(op, _):
	return [tf.reshape(op.outputs[0].ST_grad, tf.shape(op.inputs[0])), tf.zeros([], dtype=tf.int32)]

def SlstmLayer(incoming, seq_length, input_dim, output_dim, policy,
		dropout_keepprob = 0.5, pooling = False, update = "straight", scope = None, name = "SlstmLayer"):
	'''
	incomming:
		[batch_size, sen_length, choose_length, input_dim[0]]
		[batch_size, sen_length, choose_length, input_dim[1]]
	output:
		[batch_size, output_dim[0]]
		[batch_size, sen_length]
	'''
	
	with tf.variable_scope(scope, default_name=name, values=[incoming]) as scope:
		name = scope.name
		
		batch_size = tf.shape(incoming[0])[0]
		sen_length = incoming[0].get_shape()[1].value
		choose_length = incoming[0].get_shape()[2].value
		
		cell = BasicLSTMCell(output_dim[0])
		def call_cell(inputs, status):
			with tf.variable_scope("cell") as scope:
				ans = cell(inputs, status, scope = scope)[1]
				cell.reuse = True
				return ans
				
		cell_p = BasicLSTMCell(output_dim[1])
		def call_cell_p(inputs, status):
			with tf.variable_scope("cell_p") as scope:
				ans = cell_p(inputs, status, scope = scope)[1]
				cell_p.reuse = True
				return ans
		
		x_seq = tf.unstack(incoming[0], axis = 1)
		x_p_seq = tf.unstack(incoming[1], axis = 1)
		
		h_seq = [tf.zeros([batch_size, output_dim[0]]) for i in range(choose_length)]
		c_seq = [tf.zeros([batch_size, output_dim[0]]) for i in range(choose_length)]
		
		h_p_seq = [tf.zeros([batch_size, output_dim[1]]) for i in range(choose_length)]
		c_p_seq = [tf.zeros([batch_size, output_dim[1]]) for i in range(choose_length)]
		
		action_continous_seq = []
		action_seq = []
		
		for time, (x, x_p) in enumerate(zip(x_seq, x_p_seq)):
			h_pre = tf.stack(h_seq[:-choose_length-1:-1], axis = 1)
			h_pre_s = tf.stop_gradient(h_pre)
			
			h_p_pre = tf.stack(h_p_seq[:-choose_length-1:-1], axis = 1)
			
			x_s = tf.stop_gradient(x)
			
			action = policy(tf.concat([h_pre_s, h_p_pre], axis = 2), 
							tf.concat([x_s, x_p], axis = 2), scope = scope)
			
			action_continous_seq.append(action)
			
			g = tf.get_default_graph()
			with ops.name_scope("MultinomialSample") as name:
				with g.gradient_override_map({"OneHot": "ST_OneHot", "Multinomial": "ST_Multinomial"}):
					action = tf.one_hot(tf.multinomial(action, 1), choose_length, on_value = 1.0, off_value = 0.0, dtype=tf.float32, axis = -1)
			
			action_seq.append(action)
			
			h_pre = h_seq[:-choose_length-1:-1]
			c_pre = c_seq[:-choose_length-1:-1]
			h_p_pre = h_p_seq[:-choose_length-1:-1]
			c_p_pre = c_p_seq[:-choose_length-1:-1]
			x_seq = tf.unstack(x, axis = 1)
			x_p_seq = tf.unstack(x_p, axis = 1)
			all_h = []
			all_c = []
			all_h_p = []
			all_c_p = []
			for h, c, h_p, c_p, r, r_p in zip(h_pre, c_pre, h_p_pre, c_p_pre, x_seq, x_p_seq):
				state = call_cell(r, (c, h))
				all_c.append(state[0])
				all_h.append(state[1])
				state_p = call_cell_p(tf.concat([tf.stop_gradient(r), r_p], axis = 1), (c_p, h_p))
				all_c_p.append(state_p[0])
				all_h_p.append(state_p[1])
			
			action = tf.reshape(action, [batch_size, choose_length, 1])
			now_h = tf.reduce_sum(tf.stack(all_h, axis = 1) * action, axis = 1)
			now_h = tf.where(tf.less(time, seq_length), now_h, h_seq[-1]))
			now_c = tf.reduce_sum(tf.stack(all_c, axis = 1) * action, axis = 1)
			now_c = tf.where(tf.less(time, seq_length), now_c, c_seq[-1])
			now_h_p = tf.reduce_sum(tf.stack(all_h_p, axis = 1) * action, axis = 1)
			now_h_p = tf.where(tf.less(time, seq_length), now_h_p, h_p_seq[-1]))
			now_c_p = tf.reduce_sum(tf.stack(all_c_p, axis = 1) * action, axis = 1)
			now_c_p = tf.where(tf.less(time, seq_length), now_c_p, c_p_seq[-1])
			
			h_seq.append(now_h)
			c_seq.append(now_c)
			h_p_seq.append(now_h_p)
			c_p_seq.append(now_c_p)
			
		if pooling:
			output_h = tf.reduce_max(tf.stack(h_seq, axis = 2), axis = 2)
		else:
			output_h = h_seq[-1]
		
		output_h = tflearn.dropout(output_h, dropout_keepprob, name="dropOut")
		output_action = tf.stack(action_seq, axis = 1)
		action_continous = tf.stack(action_continous_seq, axis = 1)
		
	tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, output_h)
	tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, output_action)
	tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, action_continous)
	
	return output_h, output_action
	
class separate_policy():
	def __init__(self, dim, activation='prelu', keepdrop = 0.8, reuse = None):
		self.dim = dim
		self.activation = activation
		self.reuse = reuse
		self.keepdrop = keepdrop
		
	def __call__(self, h, r, scope = None, name = 'separatePolicy'):
		with tf.variable_scope(scope, default_name=name, values=[h, r], reuse = self.reuse) as scope:
			name = scope.name
			
			incoming = tf.concat([h, r], axis = 2)
			
			batch_size = tf.shape(incoming)[0]
			choose_num = incoming.get_shape()[1].value
			feature_num = incoming.get_shape()[2].value
			
			inference = tf.reshape(incoming, [batch_size * choose_num, feature_num])
			
			for d in self.dim:
				with tf.variable_scope("W" + str(d)) as scope:
					inference = tflearn.fully_connected(inference, d, activation = self.activation)
					inference = tflearn.dropout(inference, self.keepdrop)
			with tf.variable_scope("W1") as scope:
				inference = tflearn.fully_connected(inference, 1)
			
			inference = tf.reshape(inference, [batch_size, choose_num])
			inference = tf.nn.softmax(inference)
		
		self.reuse = True
		return inference