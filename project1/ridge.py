
import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    # YOUR CODE HERE
    #too slow
    # print("start here")
    # loss =0
    # #print(len(xTr[0]))
    # for i in range(len(xTr[0])):
    #     #b=xTr[:,i]
    #     #print(i,w.T)
    #     #a=w.T.dot(xTr[:,i])
    #     #print(a)
    #     #c=yTr[:,i]
    #     loss =loss+(w.T.dot(xTr[:,i])-yTr[:,i]).dot((w.T.dot(xTr[:,i])-yTr[:,i]).T)
    # loss += lambdaa *np.linalg.norm(w)**2
    # #print (loss);
    # gradient=np.zeros((len(xTr),1))
    # for i in range(len(xTr[0])):
    #     #a=(w.T.dot(xTr[:,i])-yTr[:,i])
    #     #b=xTr[:,i]
    #     d = 2*xTr[:,i]*(w.T.dot(xTr[:,i])-yTr[:,i])
    #
    #     for j in range(len(xTr)):
    #         gradient[j][0] += d[j]
    # #print(gradient)
    # #p = 2*lambdaa*w.T
    #
    # gradient+=2*lambdaa*w

    loss = (w.T.dot(xTr)-yTr).dot((w.T.dot(xTr)-yTr).T)
    #print (loss)
    loss+=lambdaa*np.linalg.norm(w)**2
    #test1=w.T.dot(xTr)-yTr
    gradient = 2*(w.T.dot(xTr)-yTr).dot(xTr.T)
    gradient+= 2*lambdaa*w.T
    #print (gradient)

    return loss,gradient.T
