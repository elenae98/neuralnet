""" Notes:
- I got this code of the PyTorch Tutorial for training a classifier, heres the link:
  https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#
- On the site, they load in the CIFAR10 dataset, so the training part has some parts that
  depends on that, need to work on how to change those parts to work with an inputted dataset
- In the "Loading and Normalizing Data" section, I tried to show how we'd input our own data
  and convert it to torch tensors, right now I think the code would only load an image and convert
  it to a tensor, but I think we need to make a whole dataset like the code does online fro the
  CIFAR10 dataset, need ot look more into that
- ** I tested this method of loading pictured and converting to a pytorch tensor, now I think we
  need to make the whole set of training photos into one dataset(batch) to traing the network with
  the method below
- I think we need to make a dataset class to make our dataset
- I was thinking we could make all the hyperparameters variables in the begining, like num_epochs, num_classes,
  batch_size, learning_rate
"""

"""Loading and Normalizing Data"""

import torch
import torchvision
import torchvision.transforms as transforms

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

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
filter2 = 5

"""fc"""
fcin1 = output2 * 8 * 8 # product of dimension of data before fc layers
fcout1 = 120
fcin2 = fcout1
fcout2 = 84
fcin3 = fcout2
fcout3 = 10

num_epochs = 2
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
        # fc = fully connected layer (in this case, there are 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """ forward pass thorugh network, I think the order goes:
        conv1 -> relu -> Maxpool -> conv2 -> relu -> Maxpool -> fc1 -> relu -> fc2 -> sigmoid -> fc3
            can change the order of these to whatever we want
        """
        x = F.relu(self.conv1(x)) # computes the activation of first convolution
        x = self.pool(x) # Maxpooling layer
        x = F.relu(self.conv2(x)) # computes the activation of second convolution
        x = self.pool(x) # Maxpooling layer
        x = x.view(-1, 16 * 5 * 5) # reshape data to input to the input layer of the neural net
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
        loss = criterion(outputs, labels) #calculates loss
        loss.backward() # backpropagation
        optimizer.step() # update the weights

        """ Need to look more into this part, I think it just prints the results
            so it's easy to read, but I'm not sure about the specifics """
        # print statistics
        running_loss += loss.item() # I think this keeps track of the loss when triaining through each mini-batch
        if i % 2000 == 1999:    # print every 2000 mini-batches (when i == 1999+n)
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')