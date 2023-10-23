import sys
import numpy as np
import gzip 
import pickle 
np.set_printoptions(threshold=sys.maxsize)
def loadData():
    f = gzip.open('C:/Users/degeo/Downloads/ArmChess/data/mnist.pkl.gz', 'rb')
    trD, vaD, teD =pickle.load(f, encoding='latin1')
    f.close()
    return (trD, vaD, teD)


def loadWrapper():
    trD, vaD, teD = loadData()
    trainInputs = [np.reshape(x,(784,1)) for x in trD[0]] # Takes xth element of trD[0] (the NMIST array data) and converts that specific array to a numpy array size (n*n, 1)
    trainResults = [vectResult(y) for y in trD[1]] # Takes each element representing the actual result ( trd[1]) & applies vectResult to it
    trainData = zip(trainInputs, trainResults) # Combines the array for the value and the vectorised representation of the value pairwise from trainData, trainResults into a zipped tuple.
    validInputs = [np.reshape(x,(784,1)) for x in vaD[0]]
    validData = zip(validInputs, vaD[1])
    testInputs = [np.reshape(x, (784, 1)) for x in teD[0]]
    testData = zip(testInputs, teD[1])
    return (trainData, validData, testData)

def vectResult(j): # Convert  the decimal numbe representation to the point in the final layer, i.e. '5' converts to [0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 ] and returns said array output
    e = np.zeros((10,1))
    e[j] = 1.0
    return e 

loadWrapper()