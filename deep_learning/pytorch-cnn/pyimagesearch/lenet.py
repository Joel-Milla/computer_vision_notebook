from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear # fully connected layers
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax # as output layer which gives categorical cross entropy
from torch import flatten # gives the 1d list of values to the fully connected layer

'''
By creating a class and using 'self.' you are saving the variables to memory and making it...
easier to debug
'''
class LeNet(Module): #Subclass of Module
    # num channels is 3 (if rgb) or 1 (if gray)
    def __init__(self, numChannels, classes):
        super(LeNet, self).__init__() # parent constructor

        '''
        - Initialization face of instantiating the space which NN needs.
        - This initializes 20 feature maps (that are the filters) where each filter will ...
        convolve the input image. So, after here, the output of layer will be 20 feature maps.
        - The kernel size will be numChannlesx5x5. So if input is gray scale, then you will ...
        have a single kernel for each feature map. that convolves the kernel and that is the output
        - however, if receive rgb, then kernel_size=3x5x5. Where each feature map will have three...
        kernels of 5x5. Each kernel will convolve a channel of RGB and then their outputs will be ....
        summed. And that result will be the output feature map. So, still the output will be 20...
        feature maps. 
        '''
        # receives how many channels the input image has, how many ...
        # feature maps output (20), and each kernel size

        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
                            kernel_size=(5,5)) #outChannelsxinChannelsx5x5
        self.relu1 = ReLU() # allows to learn non linear relationships
        self.maxpool1 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

        # initialize second set CONV => RELU => POOL LAYER
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
                            kernel_size=(5, 5)) # 50x20x5x5 channels
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

    '''
    Initialize the flow of the data, of how it will pass
    '''
    def forward(self, x):
        pass


