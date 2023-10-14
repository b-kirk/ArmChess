import numpy as np
import gzip 
import pickle 

def loadData():
    f = gzip.open('C:/Users/degeo/Downloads/ArmChess/data/mnist.pkl.gz', 'rb')
    trainData, validData, testData = pickle.load(f)
    f.close()
    return (trainData, validData, testData)

def loadWrapper():
    trD, vaD, teD = loadData()
    trainInputs = [np.reshape(x,(784,1)) for x in trD[0]]
    trainResults = [vectResult(y) for y in trD[1]]
    trainData = zip(trainInputs, trainResults)
    validInputs = [np.reshape(x,(784,1)) for x in vaD]
    validData = zip(validInputs, vaD)
    testInputs = [np.reshape(x, (784, 1)) for x in teD[0]]
    testData = zip(testInputs, teD[1])
    return (trainData, validData, testData)

def vectResult(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e 
