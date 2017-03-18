# coding:utf-8

from main import run 

class argument():pass

args = argument()

args.name = "test"
args.screen = 1
args.dataset = "sst"
args.sentence_length = 100
args.cache = 1
args.word_vector = 'wordvector/glove.refine.txt'

args.dim_w = 300

args.dim_r = 250
args.dim_rp = 0
args.choose_num = 1

args.dim_h = 250
args.dim_hp = 0
args.policy_dim = [64]

args.dim_c = [64, 5]
#args.dim_c = [5]

args.keep_drop = 0.7


args.learning_rate = 0.01

run(args)