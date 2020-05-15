import preprocess as pre

import scipy.io as sio

bostonData = sio.loadmat('./boston.mat')
xTr = bostonData['xTr']
xTe = bostonData['xTe']

xtr,xte,u,m = pre.preprocess(xTr, xTe)
