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
'''
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self): # Initialize the Net class
        super().__init__() # Access methods of nn.Module 
        self.conv1 = nn.Conv2d(3 , 6, 5) # 3 Input channels, 6 output channels, 5x5 convolution square
        self.pool = nn.MaxPool2d(2, 2) # A pool of the maximum values of each convolution (2x2 values), moving to the next two columns (stride undefined, uses kernel size) and repeating, thus halving the number of values.
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 Input channels, 16 output channels, 5x5 convolution square again.
        self.fc1 = nn.Linear(16*5*5,120) # Takes in_features (size of input sample) , out_features (size of output sample), & generates linear.weight equal to an empty parameter tensor of size in_features*out_features
        self.fc2 = nn.Linear(120, 84) # Linear Map
        self.fc3 = nn.Linear(84, 10) # Linear Map

    def forward(self, x): # Init forward prop
        x = self.pool(F.relu(self.conv1(x))) # Applies MaxPool2d to a rectified linear value of the convolution 
        x = self.pool(F.relu(self.conv2(x))) # Applies a second convolution 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Loss Function, optimizer

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(Loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
'''