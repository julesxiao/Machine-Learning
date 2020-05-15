"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K,yTr,alphas,C):
    alphas_index = 0
    alphas_i = C/2
    for index in range(len(alphas)):
        if abs(alphas[index]-C/2)<alphas_i:
            alphas_index = index
            alphas_i = abs(alphas[index]-alphas_i)
    # a = np.multiply(yTr,alphas)
    c = K[alphas_index]
    n= c.shape[0]
    c = c.reshape(1,n)
    # b=  np.multiply(np.multiply(yTr,alphas),c)
    bias = 1/yTr[alphas_index] - np.dot(c,np.multiply(yTr,alphas))
    return bias 
    
