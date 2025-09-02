# import the necessary packages
from collections import OrderedDict
import torch.nn as nn

# returns a pytorch model neural network
# 4 - 8 - 3 architecture
def get_training_model(inFeatures=4, hiddenDim=8, nbClasses=3):
    '''
    - construct a shallow, sequential neural network
    - Order dictionary names the layers to easily debug them
    - The sequential tells that output of one layer will feed into the next
	'''
    mlp_model = nn.Sequential(OrderedDict([
        ("hidden_layer_1", nn.Linear(inFeatures, hiddenDim)),
        ("activation_1", nn.ReLU()),
        ("output_layer", nn.Linear(hiddenDim, nbClasses))
    ]))

    # return the sequential model
    return mlp_model