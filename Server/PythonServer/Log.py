import datetime
import os
import sys

dt = datetime.datetime.now()
dirPath = '/home/xumeil/log/'
logFile = dt.strftime('server_%Y%m%d_%H%M%S.log')

def log(*args):
    s = ' '.join(map(str,args))
    print(s)
    f = open(dirPath+logFile, 'a+')
    f.write(s)
    f.write('\n')
    f.close()
