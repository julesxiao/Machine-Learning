import traceback

import numpy as np
import random
import sys

from example_tests import example_tests
from genTrainFeatures import genTrainFeatures
from naivebayesPY import naivebayesPY
from naivebayesPXY import naivebayesPXY
from classifyLinear import classifyLinear
from naivebayesCL import naivebayesCL
from whoareyou import whoareyou

r,ok,s = example_tests()
print(r,"  ",ok,"  ",s)

xTr,yTr = genTrainFeatures()
w,b = naivebayesCL(xTr,yTr)
preds=classifyLinear(xTr,w,b)
trainingerror=np.sum(preds!=yTr)/(yTr.shape[1])
print(trainingerror)
# whoareyou(w,b)