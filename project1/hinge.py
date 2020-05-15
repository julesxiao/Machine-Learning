from numpy import maximum, dot
import numpy as np


def hinge(w, xTr, yTr, lambdaa):
    #
    #
    # INPUT:
    # xTr dxn matrix (each column is an input vector)
    # yTr 1xn matrix (each entry is a label)
    # lambda: regularization constant
    # w weight vector (default w=0)
    #
    # OUTPUTS:
    #
    # loss = the total loss obtained with w on xTr and yTr
    # gradient = the gradient at w

    # YOUR CODE HERE
    # w size 5*1
    # xTr size 5*100
    # yTr size 1*100
    #######Calculate loss
    middle1 = 1 - yTr * np.dot(w.transpose(), xTr)
    middle2 = np.maximum(middle1, 0)
    loss = np.sum(middle2)
    loss += lambdaa * np.dot(w.transpose(),w)
    ######Calculate gradient
    middle1 = np.maximum(0, 1 - yTr * np.dot(w.transpose(), xTr))
    middle1[middle1 > 0] = 1
    middle2 = np.dot((yTr * xTr), middle1.transpose())
    gradient = - middle2 + 2 * lambdaa * w
    return loss, gradient