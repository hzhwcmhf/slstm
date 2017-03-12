
def work(fin, fout):
    data = []
    with open(fin) as f:
        for s in f:
            rating = s[1:2]
            for i in range(5):
                s = s.replace('(%d' % i, '')
            s = s.replace(')', '')
            s = rating + ' ' + s
            for i in range(50):
                s = s.replace('  ', ' ')
            data.append(s)
    with open(fout, 'w') as f:
        f.writelines(data)

work('train.origin.txt', 'train.txt')        
work('dev.origin.txt', 'dev.txt')        
work('test.origin.txt', 'test.txt')        
