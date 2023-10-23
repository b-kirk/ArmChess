import mnistLoader2
import nielsenNetwork


trainData, validData, testData = mnistLoader2.loadWrapper()

net = nielsenNetwork.net2([784, 30, 10], cost=nielsenNetwork.crossEntropyCost)
# net.SGD(trainData, 30, 10, 0.5, lmbda= 5.0, evalData=validData, monitorEvalAcc=True,monitorEvalCost=True,monitorTrainAcc=True,monitorTrainCost=True)

print(len(net.biases[0]))