import numpy as np
import  PIL 

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
    
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z)) # 1.0 Used to specify float
    
    def feedForward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = net.sigmoid(np.dot(w,a)+b)
            


arr = [1,2,3,4,5]


for x,y in zip(net.biases, net.weights):
    print(np.random.randn(y,x))
