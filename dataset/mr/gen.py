import os
import codecs
import random
random.seed(1229)

data = []
with codecs.open('neg.txt', "r", encoding='utf-8', errors='ignore') as fdata:
    now = fdata.readlines()
    data.append(['0 ' + item for item in now])
with codecs.open('pos.txt', "r", encoding='utf-8', errors='ignore') as fdata:
    now = fdata.readlines()
    data.append(['1 ' + item for item in now])

def get_test(data, n, x):
    st, ed = len(data) * x // n, len(data) * (x+1) // n
    return data[st:ed]

def get_train(data, n, x):
    st, ed = len(data) * x // n, len(data) * (x+1) // n
    return data[:st] + data[ed:]

for i in range(10):
    train_ori = [get_train(item, 10, i) for item in data]
    test_ori = [get_test(item, 10, i) for item in data]
    
    train = []
    dev = []
    test = []
    for j in range(2):
        random.shuffle(train_ori[j])
        x = len(train_ori[j]) * 9 // 10
        train += train_ori[j][:x]
        dev += train_ori[j][x:]
        test += test_ori[j]
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    os.system('mkdir mr%s' % i)
    open('mr%s/train.txt' % i, 'w').writelines(train)
    open('mr%s/dev.txt' % i, 'w').writelines(dev)
    open('mr%s/test.txt' % i, 'w').writelines(test)

