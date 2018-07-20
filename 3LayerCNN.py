import torch
import torchvision
import torchvision.transforms as transforms

"""Dataset"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

start_time = time.clock() # Timer
classes = (0, 1)

"""Defining CNN"""

import torch.nn as nn
import torch.nn.functional as F

def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)

datadim = (3,32,32) # if numpy array

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
        self.conv1 = nn.Conv2d(input1, output1, filter1, padding=2) 
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
        """ forward pass thorugh network, I think the order goes:
        conv1 -> relu -> Maxpool -> conv2 -> relu -> Maxpool -> fc1 -> relu -> fc2 -> sigmoid -> fc3
            can change the order of these to whatever we want
        """
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
    """ I think this will change based off our data
        I think this for loop puts every image through the network
        one at a time and updates the weights
    """
    for i, data in enumerate(trainloader):
        # get the inputs
        inputs, labels = data

        """I think we can keep this"""
        # zero the parameter gradients
        optimizer.zero_grad()

        """I think we can keep this too"""
        # forward + backward + optimize
        outputs = net(inputs) # get outputs
        loss = criterion(outputs, labels) # calculates loss
        loss.backward() # backpropagation
        optimizer.step() # update the weights

        """ Need to look more into this part, I think it just prints the results
            so it's easy to read, but I'm not sure about the specifics """
        # print statistics
        running_loss += loss.item() 
        if i % 2000 == 1999:    # print every 2000 mini-batches (when i == 1999+n)
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

"""Test"""
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