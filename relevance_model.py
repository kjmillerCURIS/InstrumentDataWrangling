import os
import sys
import copy
import torch
from torch import nn
import torchvision.models as models

class RelevanceModel(nn.Module):
    def __init__(self, params):
        super(RelevanceModel, self).__init__()
        self.p = copy.deepcopy(params)

        #compute self.trust_base_value
        #regularization will force ||w_k||_2^2 <= self.trust_base_value * self.p.trust_multiplier for all k
        #also, use this to figure out how length of feature vector
        if self.p.architecture == 'ResNet50':
            full_model = models.resnet50(pretrained=True)
            readout_index = -1
        else:
            assert(False)

        readout_layer = list(full_model.children())[readout_index]
        num_feats = readout_layer.weight.shape[1] #this is int, not tensor
        mags = torch.sqrt(torch.sum(torch.square(readout_layer.weight), dim=1))
        self.trust_base_value = torch.mean(mags).item() #.item() makes it a float, not a tensor

        #now initialize the weights
        self.components = nn.Conv2d(num_feats, self.p.num_components, (1,1))

        #make sure all weights are on GPU
        self.cuda()

        #regularize the initial value, for good measure
        self.regularize()

    #force ||w_k||_2^2 <= self.trust_base_value * self.p.trust_multiplier for all k
    #noop if self.p.trust_multiplier is +inf 
    def regularize(self):
        if self.p.trust_multiplier == float('+inf'):
            return

        mags = torch.sqrt(torch.sum(torch.square(self.components.weight.data), dim=1))
        mags = torch.maximum(mags, torch.tensor(1e-8, device='cuda'))
        upper = torch.tensor(self.trust_base_value * self.p.trust_multiplier, device='cuda')
        multipliers = torch.minimum(upper / mags, torch.tensor(1.0, device='cuda'))
        multipliers = torch.unsqueeze(multipliers, 1)
        self.components.weight.data = self.components.weight.data * multipliers

    #xEmbedding should be NCHW on GPU
    #will return an NCHW on GPU where C==1 and the others are the same
    #it will be probabilities
    def forward(self, xEmbedding):
        x = self.components(xEmbedding)
        x, _ = torch.max(x, dim=1, keepdims=True)
        return nn.Sigmoid()(x)

if __name__ == '__main__':
    import numpy as np
    from relevance_config import grab_params
    my_model = RelevanceModel(grab_params('RelevanceParams16Comp1Trust'))
    xEmbedding = torch.from_numpy(np.random.randn(32, 2048, 7, 7).astype('float32')).to('cuda')
    xOut = my_model(xEmbedding)
    import pdb
    pdb.set_trace()
