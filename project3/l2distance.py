import numpy as np

"""
function D=l2distance(X,Z)

Computes the Euclidean distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""


def l2distance(X, Z) :
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'

    D = np.zeros((n, m))

    D1 = np.tile(np.sum(np.square(X), axis=0).reshape(-1, 1), m) - 2 * X.T.dot(Z) + np.repeat(
        np.sum(np.square(Z), axis=0).reshape(1, -1).T, n, axis=1).T
    D1[D1 < 0] = 0
    D1 = np.power(D1, 0.5)
   # print("D1:", D1)

    # This should be done as efficiently as possible
    # (X-Z)^2 = X^2 + Z^2 - 2XZ
    ##########step1: Get X^2, add up each column
    X2 = np.sum(np.square(X), axis=0)
    # print("X2 step1:", X2)
    # turn into column vector
    X2 = X2.reshape(-1, 1)
    # print("X2 step2:", X2)
    # make sure X2 AND Z2 are same dimension
    X2 = np.tile(X2, m)
    # print("X2:",X2)
    ##########step2: Get Z^2, add up each column
    Z2 = np.sum(np.square(Z), axis=0)
    # print("Z2 step1:",Z2)
    # turn into column vector
    Z2 = Z2.reshape(-1, 1)
    # print("Z2 step2:", Z2)
    # make sure X2 AND Z2 are same dimension
    Z2 = np.tile(Z2, n).transpose()
    # print("Z2:",Z2)
    ##########step3: Get 2XZ
    XZ2 = 2 * np.dot(X.transpose(), Z)
    # print("XZ2:",XZ2)
    D = np.subtract(np.add(X2, Z2), XZ2)
    # adjust to numerical instability
    D = np.where(D < 0, 0, D)
    ##########Final step D = sqrt((X - Z) ^ 2)
    D = np.sqrt(D)
    # print("D",D)
    return D