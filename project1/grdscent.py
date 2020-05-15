import numpy as np
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent

    # YOUR CODE HERE
    for iter in range(maxiter):
        w0loss, w0gradient = func(w0)
        s = - stepsize*w0gradient
        w = w0 + s
        wloss, wgradient = func(w)
        if (wloss > w0loss):
            if(0.5*stepsize > eps):
                stepsize = 0.5*stepsize
        else:
            stepsize = 1.01 * stepsize
        w0 = w
        if np.linalg.norm(wgradient)< tolerance:
            break
    return w