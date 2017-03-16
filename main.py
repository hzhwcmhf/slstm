# coding:utf-8

import logging
import pickle

import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

from WordLoader import WordLoader
from PhraseLayer import PhraseLayer
from DataManager import DataManager
from SlstmLayer import SlstmLayer, separate_policy
from ClassifyLayer import ClassifyLayer

def run(args):
	config = tf.ConfigProto()
	config.allow_soft_placement = True
	config.gpu_options.allow_growth = True
	tf.add_to_collection('graph_config', config)


	logging.basicConfig(
		filename = ('log/%s.log' % args.name) * (1-args.screen),
		level=logging.DEBUG,
		format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',
		datefmt='%H:%M:%S')
	logging.info('----------run------------')
	logging.info("args: " + str(args.__dict__))
	
	dm = DataManager(args.dataset)
	if args.cache == 0 :
		dm.gen_word_list()
		dm.gen_data()
		pickle.dump(dm, open("./cache/data_dump", "wb"), -1)
	else :
		dm = pickle.load(open("./cache/data_dump", "rb"))
	trainY, trainX, trainLength = dm.gen_batch('train', length = args.sentence_length)
	devY, devX, devLength = dm.gen_batch('dev', length = args.sentence_length)
	testY, testX, testLength = dm.gen_batch('test', length = args.sentence_length)
	
	logging.info('----------input data ok------------')
	
	input = tflearn.input_data([None, args.sentence_length], name="input")
	seq_length = tflearn.input_data([None], name="input_len", dtype=tf.int32)
	
	wordLoader = WordLoader()
	wordvec = wordLoader.genWordVec(args.word_vector, dm.words, args.dim_w)
	embedding = tflearn.embedding(input, input_dim=len(dm.words) + 1, output_dim = args.dim_w, weights_init = tf.constant_initializer(wordvec))
	
	phrase = PhraseLayer(embedding, input_dim = args.dim_w, output_dim = (args.dim_r, args.dim_rp), output_length = args.choose_num, activation = 'prelu', dropout_keepprob = args.keep_drop, batchNorm = True)
	
	policy = separate_policy(args.policy_dim, activation='prelu',keepdrop = args.keep_drop)
	
	hidden, action = SlstmLayer(phrase, seq_length, input_dim = (args.dim_r, args.dim_rp), output_dim = (args.dim_h, args.dim_hp), policy = policy, dropout_keepprob = args.keep_drop, pooling = False, update = "straight")
	
	#hidden = tflearn.lstm(embedding, args.dim_h, dynamic=True)
	
	predict_y = ClassifyLayer(hidden, dim = args.dim_c, keepdrop = args.keep_drop, activation='prelu')
	
	#predict_y = tf.Print(predict_y, [predict_y], summarize = 10)
	
	net = tflearn.regression(predict_y, optimizer='adagrad', learning_rate=args.learning_rate,
			loss='categorical_crossentropy')
	
	for i in tf.trainable_variables():
		print i
	
	# Training
	model = tflearn.DNN(net, tensorboard_verbose=3)
	model.fit({"input":trainX, "input_len":trainLength}, trainY, validation_set=({"input":devX, "input_len":devLength}, devY), show_metric=True,
		  batch_size=32)
		  
	# regularize 
	# normalization
	# blstm