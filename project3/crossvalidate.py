"""
INPUT:	
xTr : dxn input vectors
yTr : 1xn input labels
ktype : (linear, rbf, polynomial)
Cs   : interval of regularization constant that should be tried out
paras: interval of kernel parameters that should be tried out

Output:
bestC: best performing constant C
bestP: best performing kernel parameter
lowest_error: best performing validation error
errors: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)

Trains an SVM classifier for all combination of Cs and paras passed and identifies the best setting.
This can be implemented in many ways and will not be tested by the autograder. You should use this
to choose good parameters for the autograder performance test on test data. 
"""
import numpy as np
import math
from trainsvm import trainsvm
from sklearn.model_selection import KFold

def crossvalidate(xTr, yTr, ktype, Cs, paras):
    bestC, bestP, lowest_error = 0, 0, 0

    errors = np.zeros((len(paras),len(Cs)))

    # k-fold cv
    k = 10
    kf = KFold(n_splits=k)
    # Get training error of initial classifier
    # Train initial classifier
    for p_index in range(len(paras)):
        for c_index in range(len(Cs)):
            sum_error = 0
            for train_index, test_index in kf.split(yTr):
                #print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = np.matrix([xTr[0][train_index],xTr[1][train_index]]),  np.matrix([xTr[0][test_index],xTr[1][test_index]])
                y_train, y_test = yTr[train_index], yTr[test_index]
                svmclassify = trainsvm(X_train, y_train, Cs[c_index], ktype, paras[p_index])
                train_preds = svmclassify(X_test)
                sum_error += np.mean(train_preds != y_test)
            errors[p_index][c_index] = sum_error/k

    lowest_error = np.amin(errors)
    [bestP_index, bestC_index] = np.where(errors == lowest_error)
    #print("bestP_index,bestC_index", bestP_index,bestC_index)
    bestP = paras[bestP_index[0]]
    bestC = Cs[bestC_index[0]]

    return bestC, bestP, lowest_error, errors


    