import os
import sys
import copy
import cv2
import numpy as np
import torch
from torch import nn
import torchvision.models as models

class RelevanceEmbedder:
    
    #params should be a class from relevance_config.py
    def __init__(self, params):
        self.p = copy.deepcopy(params)
        if self.p.architecture == 'ResNet50':
            full_model = models.resnet50(pretrained=True)
            num_layers_to_chop = 2
        else:
            assert(False)

        self.embedding_model = nn.Sequential(*list(full_model.children())[:-2]).to('cuda')

    #returns avg
    def get_trust_base_value(self):
        return self.trust_base_value

    #image_shape should be (H, W) from input image (numI in __call__)
    #target_shape should be (H, W) from embedding
    def __process_bboxes(self, bboxes, image_shape, target_shape):
        gt_heatmap_full = np.zeros(image_shape, dtype='float32')
        for bbox in bboxes:
            gt_heatmap_full[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] = 1.0

        hIn = torch.tensor(gt_heatmap_full[np.newaxis, np.newaxis, :, :])
        hOut = nn.AdaptiveAvgPool2d(target_shape)
        return np.squeeze(hOut.numpy())

    #makes numI into a tensor that can be fed into self.embedding_model
    #this tensor will be put onto the GPU, so that we can do the permute() operation there
    def __image_to_tensor__(self, numI):
        #flip channels and normalize
        numI = cv2.cvtColor(numI, cv2.COLOR_BGR2RGB)
        numI = (numI / 255.0).astype('float32')
        for c in range(3):
            numI[:,:,c] = (numI[:,:,c] - self.p.color_means[c]) / self.p.color_SDs[c]
                
        #resize
        smaller_size, bigger_size = min(numI.shape[:2]), max(numI.shape[:2])
        bigger_size *= self.p.default_smaller_size / smaller_size
        bigger_size = int(round(bigger_size / self.p.size_multiple)) * self.p.size_multiple
        smaller_size = self.p.default_smaller_size
        if numI.shape[0] < numI.shape[1]: #height < width
            numI = cv2.resize(numI, (bigger_size, smaller_size), cv2.INTER_CUBIC)
        else: #width <= height
            numI = cv2.resize(numI, (smaller_size, bigger_size), cv2.INTER_CUBIC)

        #convert to tensor, put on GPU, and change dimensions
        numI = numI[np.newaxis,:,:,:]
        xImage = torch.tensor(numI).to('cuda').permute(0, 3, 1, 2)
        return xImage

    #I feel like making callable object
    #expects numI to be BGR, uint8, 0-255, a single image
    #any augmentations should already be done to it
    #will return a float32 tensor of shape NCHW, where N==1 and C==2048
    #this tensor is computed on the GPU
    #if embedding_to_CPU is true, then we do .to('cpu').detach() on it - use this during training/validation when you want to store lots of image embeddings in RAM
    #if embedding_to_CPU is false, then we only do .detach() on it - use this in testing when you want to do further inference on an embedded image, and then let go of the reference
    #if bboxes is None, then we'll only return the embedding
    #but if bboxes is not None, then it should be a list of dictionaries
    #possibly an empty list if image has no bboxes labelled
    #each bbox should be a dictionary with keys 'xmin', 'ymin', 'xmax', 'ymax'
    #it should be aligned with the image, i.e. any flip and/or rot90 augmentations should also be done to the bboxes
    #we will return a float32 numpy array of shape (H, W)
    #This might seem inconsistent with the other returned value, but it makes sense because the returned embedding might go straight into another NN, whereas the heatmap won't
    #bbox computation happens entirely on the CPU and is detached - we're just using AdaptiveAvgPool2d, so it's not worth the GPU overhead
    def __call__(self, numI, embedding_to_CPU=False, bboxes=None):
        xImage = self.__image_to_tensor__(numI)
        xEmbedding = self.embedding_model(xImage)
        if embedding_to_CPU:
            xEmbedding = xEmbedding.to('cpu')

        xEmbedding = xEmbedding.detach()

        if bboxes is None:
            return xEmbedding

        gt_heatmap = self.__process_bboxes(bboxes, numI.shape[:2], tuple(xEmbedding.shape[-2:]))
        return xEmbedding, gt_heatmap

if __name__ == '__main__':
    from relevance_config import grab_params
    my_embedder = RelevanceEmbedder(grab_params('RelevanceParams16Comp1Trust'))
    numI = np.random.randint(0, high=256, size=(224, 448, 3)).astype('uint8')
    xEmbedding = my_embedder(numI)
    import pdb
    pdb.set_trace()
