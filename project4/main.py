import pickle
import numpy as np

# run this to save edit best_parameters.pickle which will be used to determine performance on the autograder
# Also feel free to use this file to do any testing as it will not be called by the autograder

best_parameters = {
    'TRANSNAME' : 'sigmoid',
    'ROUNDS' : 50,
    'ITER' : 10,
    'STEPSIZE' : 0.08,
    'wst' : np.array([1,20,10,20,13])
}

with open('best_parameters.pickle', 'wb') as f:
    pickle.dump(best_parameters, f)