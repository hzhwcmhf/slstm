# coding:utf-8

import logging
import pickle

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences



def run(args):

	logging.basicConfig(
		filename = ('log/%s.log' % args.name) * (1-args.screen),
		level=logging.DEBUG,
		format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',
		datefmt='%H:%M:%S')
	logging.info('----------run------------')
	logging.info("args: " + str(args))
	
	dm = DataManager(args.dataset)
	if args.cache == 0 :
		dm.gen_word_list()
		dm.gen_data()
		pickle.dump(dm, open("./cache/data_dump", "wb"), -1)
	else :
		dm = pickle.load(open("./cache/data_dump", "rb"))
	logging.info('----------input data ok------------')