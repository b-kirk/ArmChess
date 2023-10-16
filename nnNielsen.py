import numpy as np
import random
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z)) # 1.0 Used to specify float
def sigmoidPrime(z):
    return sigmoid(z)*(1-sigmoid(z))
class net1():
    def __init__(self, neuronAmount):
        self.numLayers = len(neuronAmount)
        self.neuronAmount = neuronAmount
        self.biases = [np.random.randn(y,1) for y in neuronAmount[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(neuronAmount[:-1], neuronAmount[1:])]
    def feedForward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
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
        delB = [np.zeros(b.shape) for b in self.biases]
        delW = [np.zeros(w.shape) for w in self.weights]
        for x, y in miniBatch:
            deltaDelB, deltaDelW = self.backProp(x, y)
            delB = [db+ddb for db, ddb in zip(delB, deltaDelB)]
            delW = [nw+dnw for nw, dnw in zip(delW, deltaDelW)]
        self.weights = [w-(eta/len(miniBatch))*nw for w, nw in zip(self.weights, delW)]
        self.biases = [b-(eta/len(miniBatch))*nb for b, nb in zip(self.biases, delB)]
    def backProp(self, x, y):
        delB = [np.zeros(b.shape) for b in self.biases]
        delW = [np.zeros(w.shape) for w in self.weights]
        # FF
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backpass
        delta = self.costDerivative(activations[-1], y)*sigmoidPrime(zs[-1])
        delB[-1] = delta
        delW[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2,self.numLayers):
            z = zs[-l]
            sp = sigmoidPrime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta)*sp
            delB[-l] = delta
            delW[-l] = np.dot(delta, activations[-l-1])
        return (delB, delW)
    def costDerivative(self, outputActivations, y):
        return (outputActivations-y)
class CrossEntropyCost():
    @staticmethod
    def fn(a ,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    @staticmethod
    def delta(z, a, y):
        return (a-y)
class QuadraticCost():
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2
    def delta(z, a, y):
        return (a-y)*sigmoid_prime(z)
class net2():
    def __init__(self, neuronAmount, cost=crossEntropyCost):
        self.numLayers = len(neuronAmount)
        self.neuronAmount = neuronAmount
        self.defaultWeights()
        self.cost = cost
    def defaultWeights(self):
        self.biases = [np.random.randn(y, 1) for y in self.neuronAmount[1:]] 
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.neuronAmount[:-1], self.neuronAmount[1:])]
    def largeWeights(self):
        self.biases = [np.random.randn(y, 1) for y in self.neuronAmount[1:]] 
        self.weights = [np.random.randn(y, x) for x, y in zip(self.neuronAmount[:-1], self.neuronAmount[1:])]
    def feedForward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
            return a
    def SGD(self, trainData, epochs, miniBatchSize, eta, lmbda = 0.0, evalData = None, monitorEvalCost=False, monitorEvalAcc=False, monitorTrainCost=False, monitorTrainAcc=False):
        if evalData: evalDataLength = len(evalData)
            trainDataLength = len(trainData)
        evalCost, evalAcc = [], []
        trainCost, trainAcc = [], []
        for j in range(epochs):
            random.shuffle(trainData)
            miniBatches = [trainData[k:k+miniBatchSize] for k in range(0, trainDataLength, miniBatchSize)]
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta, lmbda, len(trainData))
            print("Epoch %s training complete" %j )
            if monitorTrainCost:
                cost = self.totalCost(trainData, lmbda)
                trainCost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitorTrainAcc:
                accuracy = self.accuracy(trainData, convert=True)
                trainAccuracy.append(accuracy)
                print("Accuracy on training data: {}/{}".format(accuracy, trainDataLength))
            if monitorEvalCost:
                cost = self.totalCost(evalData, lmbda, convert=True)
                evalCost.append(cost)
                print("Cost on evaluaton data: {}".format(cost))
            if monitorEvalAcc:
                accuracy = self.accuracy(evalData)
                evalAccuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(accuracy, evalDataLength))
            print
        return evalCost, evalAccuracy, trainCost, trainAccuracy
    def updateMiniBatch(self, miniBatch, eta, lmbda, n):
        delB = [np.zeros(b.shape) for b in self.biases]
        delW = [np.zeros(w.shape) for w in self.weights]
        for x, y in miniBatch:
            deltaDelB, deltaDelW = self.backProp(x, y)
            delB = [db+ddb for db, ddb in zip(delB, deltaDelB)]
            delW = [nw+dnw for nw, dnw in zip(delW, deltaDelW)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(miniBatch))*nw for w, nw in zip(self.weights, delW)]
        self.biases = [b-(eta/len(miniBatch))*nb for b, nb in zip(self.biases, delB)]
    def backProp(self, x, y):
        delB = [np.zeros(b.shape) for b in self.biases]
        delW = [np.zeros(w.shape) for w in self.weights]
        # FF
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backpass
            
    
        
