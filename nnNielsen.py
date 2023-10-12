import numpy as np
import random
import  PIL 

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z)) # 1.0 Used to specify float

class net(object):
    '''
    Creates a Neural Network layout, parameters specifying the amount of neurons in each layer. Example: NeuralNetwork(1,2,3) would have one neuron in the first layer, two in the second, and three in the third.

    '''
    public:

    def __init__(self, neuronAmount):
        self.numLayers = len(neuronAmount)
        self.neuronAmount = neuronAmount
        self.biases = [np.random.randn(y,1) for y in neuronAmount[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(neuronAmount[:-1], neuronAmount[1:])]
    
    
    def feedForward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = net.sigmoid(np.dot(w,a)+b)
            return a
    def SGD(self, trainData, epochs, miniBatchSize , eta, testData = None):
        if testData: testDataLength = len(testData)
        n = len(trainData)
        for j in range(epochs):
            random.shuffle(trainData)
            miniBatches = [ trainData[k:k+miniBatchSize] for k in range(0, n, miniBatchSize)]
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta)
            if testData:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(testData), testDataLength))
            else:
                print("Epoch {0} complete".format(j))
    def updateMiniBatch(self, miniBatch, eta):
        """Apply gradient descent via backprop to a mini-batch (a list of tuples, "x, y"). Eta is the learning rate """
        delB = [np.zeros(b.shape) for b in self.biases]
        delW = [np.zeros(w.shape) for w in self.weights]
        for x, y in miniBatch:
            deltaDelB, deltaDelW = self.backProp(x, y)
            delB = [db+ddb for db, ddb in zip(delB, deltaDelB)]
            delW = [nw+dnw for nw, dnw in zip(delW, deltaDelW)]
        self.weights = [w-(eta/len(miniBatch))*nw for w, nw in zip(self.weights, delW)]
        self.biases = [b-(eta/len(miniBatch))*nb for b, nb in zip(self.biases, delB)]



