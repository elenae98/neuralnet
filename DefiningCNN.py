import torch
import torch.nn as nn
import torch.nn.functional as F

""" Notes:
- to my understanding, there are two sectons to the code: (1) creating the NN, and (2) training the NN
  (there could be more, not sure yet)
- the parameters (inputs to nn.Conv2d) probably depends on the dimensions of the input data
- I wrote in some commetns to based off the code in https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/
- 
"""

def outputSize(in_size, kernel_size, stride, padding):
    """ From website
        function that tells you the output size of a dimension in each part of the process based off
        the input size, kernal size, stride, and padding
        (look at website for specifics) 
    """
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel (b/c greyscale, 3 if RGB), 6 output channels (# of nodes in the next layer),
        # 5x5 square convolution kernel(filter)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # fc = fully connected layer (in this case, there are 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """ Forward pass thorugh the Network
            I think this means that the layers it is going through is:
            conv1 -> relu -> max_pool -> conv2 -> relu -> max_pool -> fc1 -> fc2 -> fc3
        """
        # Max pooling over a (2, 2) window
        # ORIGIONAL CODE x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # I think is the same as
        x = F.relu(self.conv1(x)) # computes the activation of first convolution
        x = F.max_pool2d(x, (2,2))
        # If the size is a square you can only specify a single number
        # ORIGIONAL CODE x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # I think is the same as
        x = F.relu(self.conv2(x)) # computes the activation of second convolution
        x = F.max_pool2d(x, 2)
        # reshape data to input to the input layer of the neural net
        # size changes from whatever it was to (1, self.num_flat_features(x)) in this case self.num_flat_features(x) = 16 * 5 * 5
        x = x.view(-1, self.num_flat_features(x))
        # computes the activation of the first fully connected layer
        x = F.relu(self.fc1(x)) # size changes from (1, 16 * 5 * 5) to (1, 120)
        # computes the activation of the second fully connected layer
        x = F.relu(self.fc2(x)) # size changes from (1, 120) to (1, 84)
        # computes the third fully connected layer (activation applied later?? *from breakdown on website*)
        x = self.fc3(x) # size changes from (1, 84) to (1, 10)
        return x

    def num_flat_features(self, x):
        """ product of dimensions except the batch dimension
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)


