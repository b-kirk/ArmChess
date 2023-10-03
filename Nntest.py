import numpy as np
import torch 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data

'''
image = Image.open('C:/Users/degeo/Downloads/ArmChess/images/pom.png')
data = np.asarray(image)
print("this is a uhhhh shape." , data.shape)
# Converts 2D PIL Matrix of pixel values into 1D array
n_k = data.ravel()
nArray = np.array(n_k)
# Convert to PyTorch GPU compatible array 
nGPUArray = torch.from_numpy(nArray)
print(nGPUArray)
'''
# Implementation of the PyTorch 'Deep Learning with PyTorch: A 60 Minute Blitz '
# Loading CIFAR10 data for image recognition

# Normalization function

trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batchSize = 4

trainBatch = torchvision.datasets.CIFAR10(root= './data', train=True, download=True, transform=trans)

Loader = torch.utils.data.DataLoader(trainBatch, batch_size=batchSize, shuffle=True, num_workers=0)

testData = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans)

testLoad = torch.utils.data.DataLoader(testData, batch_size=batchSize, shuffle=False, num_workers=2)

items = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def showImage(img):
    img = img/2 + 0.5 # Reverses normalization to show an image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

iterate = iter(Loader)
images, labels = next(iterate)


# Displays a random selection of batchSize worth of the image dataset

showImage(torchvision.utils.make_grid(images))

print(''.join(f'{items[labels[j]]:5s}' for j in range(batchSize)))

# Convolutional Neural Network 

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self): # Initialize the Net class and assigns values to
        super().__init__() # Access methods of nn.Module 
        self.conv1 = nn.Conv2d(3 , 6, 5) # 3 Input channels, 6 output channels, 5x5 convolution square
        self.pool = nn.MaxPool2d(2, 2) # A pool of the maximum values of each convolution (2x2 values), moving to the next two columns (stride undefined, uses kernel size) and repeating, thus halving the number of values.
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 Input channels, 16 output channels, 5x5 convolution square again.
        self.fc1 = nn.Linear(16*5*5,120) # Linear transform 
        self.fc2 = nn.Linear(120, 84) # 
        self.fc3 = nn.Linear(84, 10)
