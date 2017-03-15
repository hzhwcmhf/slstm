# coding:utf-8

from main import run 

class argument():pass

args = argument()

args.name = "test"
args.screen = 1
args.dataset = "sst"
args.sentence_length = 100

args.dim_w = 300

args.dim_r = 250
args.dim_rp = 50
args.choose_num = 3

args.policy_dim = [64]

args.dim_c1 = [64, 5]

args.keep_drop = 0.7


args.learning_rate = 0.001

run(args)