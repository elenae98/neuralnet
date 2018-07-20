

import torch
import torchvision
import torchvision.transforms as transforms

"""Dataset"""
           # Create two dictionaries, one containing the IDs of images, and other containing labels of images
imageIDs = {'trainingData': ['D-1-B-N-001-001.map', 'D-1-B-N-001-002.map', 'D-1-B-N-001-003.map', 'D-1-B-N-001-004.map', 'D-1-B-N-001-005.map', 'D-1-B-N-001.map', 'D-0-A-N-001.map', 'D-0-A-N-002.map', 'D-0-A-N-003.map', 'D-0-A-N-004.map', 'D-0-A-N-005.map', 'D-0-A-Y-001-000.map', 'D-0-A-Y-001-001.map', 'D-0-A-Y-001-002.map', 'D-0-A-Y-001-003.map', 'D-0-A-Y-001-004.map', 'D-0-A-Y-001-005.map', 'D-0-A-Y-001-006.map', 'D-0-A-Y-001-007.map', 'D-0-A-Y-001-008.map', 'D-0-A-Y-001-009.map', 'D-0-A-Y-001-010.map', 'D-0-A-Y-001-011.map', 'D-0-A-Y-002-000.map', 'D-0-A-Y-002-001.map', 'D-0-A-Y-002-002.map', 'D-0-A-Y-002-003.map', 'D-0-A-Y-002-004.map', 'D-0-A-Y-002-005.map', 'D-0-A-Y-002-006.map', 'D-0-A-Y-002-007.map', 'D-0-A-Y-002-008.map', 'D-0-A-Y-002-009.map', 'D-0-A-Y-002-010.map', 'D-0-A-Y-002-011.map', 'D-0-A-Y-003-000.map', 'D-0-A-Y-003-001.map', 'D-0-A-Y-003-002.map',  'D-0-A-Y-003-003.map', 'D-0-A-Y-003-004.map', 'D-0-A-Y-003-005.map', 'D-0-A-Y-003-006.map', 'D-0-A-Y-003-007.map', 'D-0-A-Y-003-008.map', 'D-0-A-Y-003-009.map', 'D-0-A-Y-003-010.map', 'D-0-A-Y-003-011.map'], 'testData': [...]}
trainingLabels = {}

# We can fill labels dictionary using for loop and by making the names of the images something like:
# Torn_1, Torn_2, ..., notTorn_1, notTorn_2, ... (like above)
for ID in imageIDs['trainingData']:
    if ID[2] == '0':
        trainingLabels[ID] = 0 #clean
    else:
        trainingLabels[ID] = 1 #problematic

trainingDataset = TrainingSet(imageIDs['trainingData'], trainingLabels, '/data')

trainingLoader = torch.utils.data.DataLoader(trainingDataset, batch_size=1,
                                          shuffle=True, num_workers=0) # ask about variables (batch_size, num_workers, etc)

"""Defining CNN"""

import torch.nn as nn
import torch.nn.functional as F

def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)

datadim = (3,2048,2592) # if numpy array (check if array tensor)

# aks john about kernal_size and stride in relation to data size
# also ask john about inputnodes/outputnodes

"""conv1"""
input1 = datadim[0]
output1 = 6
filter1 = 5

"""conv2"""
input2 = output1
output2 = 16 
filter2 = 3

"""conv3"""
input3 = output2
output3 = 32 
filter3 = filter2

"""fc"""
fcin1 = output3 * 5 * 5 # product of dimension of data before fc layers
fcout1 = 150
fcin2 = fcout1
fcout2 = 90
fcin3 = fcout2
fcout3 = len(classes)

num_epochs = 4
learning_rate = 0.001

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        """ defining convalutional layers, pooling layers, and fully connected layers
            probably need to change numbers based off input data
            I think we define the layers we want to use here
        """
        

        # 1 input image channel (b/c greyscale, 3 if RGB)
        # 6 output channels (# of nodes in the next layer)
        # 5x5 square convolution kernel(filter)
        self.conv1 = nn.Conv2d(input1, output1, filter1) 
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        # 6 input channels/nodes
        # 16 output channels/nodes (# of nodes in the next layer)
        # 5x5 square convolution kernel(filter)
        self.conv2 = nn.Conv2d(input2, output2, filter2)
        self.conv3 = nn.Conv2d(input3, output3, filter3)
        # fc = fully connected layer (in this case, there are 3)
        self.fc1 = nn.Linear(fcin1, fcout1)
        self.fc2 = nn.Linear(fcin2, fcout2)
        self.fc3 = nn.Linear(fcin3, fcout3)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # computes the activation of first convolution
        x = self.pool(x) # Maxpooling layer
        x = F.relu(self.conv2(x)) # computes the activation of second convolution
        x = self.pool(x) # Maxpooling layer
        x = F.relu(self.conv3(x)) # computes the activation of second convolution
        x = x.view(-1, 32 * 5 * 5) # reshape data to input to the input layer of the neural net
        x = F.relu(self.fc1(x)) # computes the activation of the first fully connected layer
        x = F.logsigmoid(self.fc2(x)) # computes the activation of the second fully connected layer
        x = self.fc3(x) # computes the third fully connected layer
        return x


net = Net()

"""Defining Loss Function and Optimizer"""

import torch.optim as optim

criterion = nn.CrossEntropyLoss() # basically the loss
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9) # helps adjust the weights when training

"""Training the CNN"""

for epoch in range(num_epochs):  # loop over the dataset multiple times
                        # epoch is once over the full batch
    running_loss = 0.0

    for i, data in enumerate(trainingLoader):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) # get outputs
        loss = criterion(outputs, labels) # calculates loss
        loss.backward() # backpropagation
        optimizer.step() # update the weights

        # print statistics
        running_loss += loss.item() 
        if i % 2000 == 1999:    # print every 2000 mini-batches (when i == 1999+n) --> need to make sure the numbers work
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

"""Test""" 
# we need a testing set
# we will need to edit the printing for our dataset
import matplotlib.pyplot as plt
import numpy as np

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(' ')
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
print(' ')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

print("--- %s seconds ---" % (time.clock() - start_time)) # Print time elapsed
    
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

"""
#get random images
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
"""
