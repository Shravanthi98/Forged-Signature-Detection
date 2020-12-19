# Referred from https://pytorch.org/docs/stable/index.html
# import necessary packages
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout


# Model construction
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining 1st 2D convolution layer
            Conv2d(1, 96, kernel_size=11, stride=4, padding=0),
            BatchNorm2d(96),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            # Defining 2nd 2D convolution layer
            Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            # Defining 3rd 2D convolution layer
            Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(384),
            ReLU(),
            # Defining 4th 2D convolution layer
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(384),
            ReLU(),
            # Defining 5th 2D convolution layer
            Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
        )
        # Defining the Linear layers
        self.linear_layers = Sequential(
            Dropout(p=0.5, inplace=True),
            Linear(9216, 4096),
            ReLU(),
            Dropout(p=0.5, inplace=True),
            Linear(4096, 4096),
            ReLU(),
            Linear(4096, 2),

        )
        self.init_bias()  # initialize bias

    # Initializing the weights and biases
    def init_bias(self):
        for layer in self.cnn_layers:
            if isinstance(layer, Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.cnn_layers[4].bias, 1)
        nn.init.constant_(self.cnn_layers[11].bias, 1)
        nn.init.constant_(self.cnn_layers[14].bias, 1)

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.linear_layers(x)
        return x


if __name__ == '__main__':
    print('AlexNet created')
