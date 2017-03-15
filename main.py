# coding:utf-8

import logging
import pickle


import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

from WordLoader import WordLoader
from PhraseLayer import PhraseLayer
from DataManager import DataManager
from SlstmLayer import SlstmLayer
from ClassifyLayer import ClassifyLayer

def run(args):

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
	trainY, trainX = dm.gen_batch('train', length = args.sentence_length)
	devY, devX = dm.gen_batch('dev', length = args.sentence_length)
	testY, testX = dm.gen_batch('test', length = args.sentence_length)
	
	logging.info('----------input data ok------------')
	
	input = tflearn.input_data([None, args.sentence_length])
	
	wordLoader = WordLoader()
	wordvec = wordLoader.genWordVec(args.word_vector, dm.words, args.dim_w)
	embedding = tflearn.embedding(input, weights_init = wordvec)
	
	phrase = PhraseLayer(embedding, input_dim = args.dim_w, output_dim = (args.dim_r, args.dim_rp), output_length = args.choose_num, activation = 'prelu', dropout_keepprob = args.keep_drop, batchNorm = True)
	
	policy = separate_policy(args.policy_dim, activation='prelu',keepdrop = args.keep_drop, reuse = True)
	
	hidden, action = SlstmLayer(phrase, input_dim = (args.dim_r, args.dim_rp), output_dim = (args.dim_h, args.dim_hp), policy = policy, dropout_keepprob = args.keep_drop, pooling = True, update = "straight")
	
	predict_y = ClassifyLayer(hidden, dim = args.dim_c, keepdrop = args.keep_drop, activation='prelu')
	
	net = tflearn.regression(predict_y, optimizer='adam', learning_rate=args.learning_rate,
			loss='categorical_crossentropy')
						 
	# Training
	model = tflearn.DNN(net, tensorboard_verbose=0)
	model.fit(devX, devY, validation_set=(devX, devY), show_metric=True,
		  batch_size=32)
		  
	# regularize 
	# normalization
	# blstm