import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
    # w size 5*1
    # xTr size 5*100
    # yTr size 1*100
def logistic(w,xTr,yTr):

    #YOUR CODE HERE

    loss = np.sum(np.log(1 + np.exp(- yTr * np.dot(w.transpose(),xTr))))
    #print(loss)
    # middle1 = yTr * np.dot(w.transpose(),xTr)
    # middle2 = np.exp(middle1)+1
    middle3 = np.sum(np.divide(-(yTr * xTr), np.exp(yTr * np.dot(w.transpose(),xTr))+1),axis = 1)
    # gradient = w;
    # for i in range(len(w)):
    #     gradient[i] = middle3[i]
    # print(gradient)
    # print (loss)
    gradient = middle3.reshape(middle3.shape[0], 1)

    return loss,gradient