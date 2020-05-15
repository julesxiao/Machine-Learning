

import numpy as np

from createsvmclassifier import createsvmclassifier
from generateQP import generateQP
from recoverBias import recoverBias
from trainsvm import trainsvm
from crossvalidate import crossvalidate
import matplotlib.pyplot as plt
from visdecision import visdecision
import pickle
import scipy.io as sio
import sys
from computeK import computeK
from cvxopt import solvers
x = np.genfromtxt('xTr.csv', delimiter=',')
y = np.genfromtxt('yTr.csv', delimiter=',').reshape((x.shape[1], 1))


k= computeK('linear',x,x,1)


q,p,g,h,a,b =generateQP(k,y,1)
sol = solvers.qp(q,p,g,h,a,b)
#print(sol['x'])
alpha = sol['x']
bias = recoverBias(k,y,alpha,1)
print(bias)
# prediction = createsvmclassifier(x, y, alpha, bias, 'linear', 1)
# print (prediction)
# a = np.matrix([[1, 2], [3, 4]])
# b=np.matrix([[1, 2], [3, 4]])
# k = a*b
# k2 = np.dot(a,b)
# k3 = np.multiply(a,b)
# print(k)
# print(k2)
# print(k3)


