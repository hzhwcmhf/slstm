# coding:utf-8

import tflearn
from tflearn import variables as va
from tflearn.layers.recurrent import BasicLSTMCell

@ops.RegisterGradient("ST_OneHot")
def _ST_OneHot(op, grad):
	op.inputs[0].ST_grad = grad
	return [tf.zeros(op.inputs[0].get_shape()), 0, 0, 0]

@ops.RegisterGradient("ST_Multinomial")
def _ST_Multinomial(op, _):
	return [op.outputs[0].ST_grad, 0]

def SlstmLayer(incoming, input_dim, output_dim, policy,
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
		
		seq_length = incoming[0].seq_length
		
		cell = BasicLSTMCell(output_dim[0], reuse = True)
		call_cell = lambda inputs, status: cell(scope = tf.variable_scope(scope, default_name="cell"))[1]
		cell_p = BasicLSTMCell(output_dim[1], reuse = True)
		call_cell_p = lambda inputs, status: cell(scope = tf.variable_scope(scope, default_name="cell_p"))[1]
		
		x_seq = tf.unstack(incoming[0], axis = 1)
		x_p_seq = tf.unstack(incoming[1], axis = 1)
		
		h_seq = [tf.zeros((batch_size, outdim[0])) for i in range(choose_length)]
		c_seq = [tf.zeros((batch_size, outdim[0])) for i in range(choose_length)]
		
		h_p_seq = [tf.zeros((batch_size, outdim[1])) for i in range(choose_length)]
		c_p_seq = [tf.zeros((batch_size, outdim[1])) for i in range(choose_length)]
		
		action_continous_seq = []
		action_seq = []
		
		for time, (x, x_p) in enumerate(zip(x_seq, x_p_seq)):
			h_pre = tf.pack(h_seq[:-choose_length-1:-1], axis = 1)
			h_pre_s = tf.stop_gradient(h_pre)
			
			h_p_pre = tf.pack(h_p_seq[:-choose_length-1:-1], axis = 1)
			
			x_s = tf.stop_gradient(x)
			
			action = policy(tf.concat([h_pre_s, h_p_pre], axis = 2), 
							tf.concat([x_s, x_p], axis = 2), scope = scope)
			
			action_continous_seq.append(action)
			
			with ops.name_scope("MultinomialSample") as name:
				with g.gradient_override_map({"OneHot": "ST_OneHot", "Multinomial": "ST_Multinomial"}):
					action = tf.one_hot(tf.multinomial(action, 1), choose_length, on_value = 1, 
						off_value = 0, axis = -1)
			
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
				state_p = call_cell_p(tf.concat(tf.stop_gradient(r), r_p, axis = 1), (c_p, h_p))
				all_c_p.append(state_p[0])
				all_h_p.append(state_p[1])
			
			now_h = tf.reduce_sum(tf.pack(all_h, axis = 1) * action, axis = 1)
			now_h = tf.where(tf.less(time, incoming[0].seq_length), now_h, tf.zeros([batch_num, output_dim[0]]))
			now_c = tf.reduce_sum(tf.pack(all_c, axis = 1) * action, axis = 1)
			now_c = tf.where(tf.less(time, incoming[0].seq_length), now_c, tf.zeros([batch_num, output_dim[0]]))
			now_h_p = tf.reduce_sum(tf.pack(all_h_p, axis = 1) * action, axis = 1)
			now_h_p = tf.where(tf.less(time, incoming[0].seq_length), now_h_p, tf.zeros([batch_num, output_dim[1]]))
			now_c_p = tf.reduce_sum(tf.pack(all_c_p, axis = 1) * action, axis = 1)
			now_c_p = tf.where(tf.less(time, incoming[0].seq_length), now_c_p, tf.zeros([batch_num, output_dim[1]]))
			
			h_seq.append(now_h)
			c_seq.append(now_c)
			h_p_seq.append(now_h_p)
			c_p_seq.append(now_c_p)
			
		if pooling:
			output_h = tf.reduce_max(tf.pack(h_seq, axis = 2), axis = 2)
		else:
			output_h = h_seq[-1]
		
		output_h = tf.dropout(output_h, dropout_keepprop, name="dropOut")
		output_action = tf.pack(action_seq, axis = 1)
		action_continous = tf.pack(action_continous_seq, axis = 1)
		
	tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, output_h)
	tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, output_action)
	tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, action_continous)
	
	return output_h, output_action
	
class seperate_policy():
	def __init__(self, dim, activation='prelu', keepdrop = 0.8, reuse = False):
		self.dim = dim
		self.activation = activation
		self.reuse = reuse
		self.keepdrop = keepdrop
		
	def __call__(h, r, scope = None, name = 'separatePolicy')
		with tf.variable_scope(scope, default_name=name, values=[h, r]) as scope:
			name = scope.name
			
			batch_num = h.get_shape()[0].value
			choose_num = h.get_shape()[1].value
			
			incoming = tf.concat([h, r], axis = 2)
			
			inference = tf.reshape(incoming, [batch_num * choose_num, -1])
			
			for d in self.dim:
				inference = tflearn.fully_connected(inference, d, activation = self.activation, reuse = self.reuse, scope = scope)
				inference = tflearn.dropout(inference, self.keepdrop)
			inference = tflearn.fully_connected(inference, 1, reuse = self.reuse, scope = scope)
			
			inference = tf.reshape(reference, [batch_num, -1])
			inference = tf.softmax(reference)
			
		return inference