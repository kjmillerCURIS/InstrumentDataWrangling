import os
import sys
import copy
import cv2
import numpy as np
import random
from tqdm import tqdm
from relevance_embedder import RelevanceEmbedder

#returns numI_perm, bboxes_perm
def permute_image_and_bboxes(numI, bboxes, perm):
    assert(isinstance(perm, int) and perm >= 0 and perm < 8)
    numI_perm = copy.deepcopy(numI)
    bboxes_perm = copy.deepcopy(bboxes)
    
    #check each bit of perm to decide whether to do vertical flip, horizontal flip, transpose
    #important to permute bboxes BEFORE numI so we can use numI.shape properly
    if perm & 1: #vertical flip
        h = numI_perm.shape[0]
        new_bboxes_perm = []
        for bbox in bboxes_perm:
            new_bbox={'xmin':bbox['xmin'],'xmax':bbox['xmax'],'ymin':h-bbox['ymax']-1,'ymax':h-bbox['ymin']-1}
            new_bboxes_perm.append(new_bbox)

        bboxes_perm = new_bboxes_perm
        numI_perm = np.flipud(numI_perm)
    
    if perm & 2: #horizontal flip
        w = numI_perm.shape[1]
        new_bboxes_perm = []
        for bbox in bboxes_perm:
            new_bbox={'ymin':bbox['ymin'],'ymax':bbox['ymax'],'xmin':w-bbox['xmax']-1,'xmax':w-bbox['xmin']-1}
            new_bboxes_perm.append(new_bbox)

        bboxes_perm = new_bboxes_perm
        numI_perm = np.fliplr(numI_perm)
    
    if perm & 3: #translation
        new_bboxes_perm = []
        for bbox in bboxes_perm:
            new_bbox={'xmin':bbox['ymin'],'xmax':bbox['ymax'],'ymin':bbox['xmin'],'ymax':bbox['xmax']}
            new_bboxes_perm.append(new_bbox)

        bboxes_perm = new_bboxes_perm
        numI_perm = np.transpose(numI_perm, (1, 0, 2))

    return np.ascontiguousarray(numI_perm), bboxes_perm

def check_inputs(image_filenames, bboxes_dict):
    seen = set([])
    for image_filename in image_filenames:
        image_base = os.path.splitext(os.path.basename(image_filename))[0]
        assert(image_base not in seen)
        assert(image_base in bboxes_dict)
        seen.add(image_base)

#return xEmbedding_dict, prob_dict
def run_embedder(image_filenames, bboxes_dict, mode, params):
    p = params
    check_inputs(image_filenames, bboxes_dict)
    re = RelevanceEmbedder(p) #important to define this here and not in the main generator function, so it can go out of scope and out of GPU memory
    xEmbedding_dict = {}
    prob_dict = {}
    for image_filename in tqdm(image_filenames):
        image_base = os.path.splitext(os.path.basename(image_filename))[0]
        numI = cv2.imread(image_filename)
        bboxes = bboxes_dict[image_base] 
        perm_list = {'train' : range(8), 'val' : [0]}
        for perm in perm_list[mode]:
            numI_perm, bboxes_perm = permute_image_and_bboxes(numI, bboxes, perm)
            xEmbedding, prob_map = re(numI_perm, embedding_to_CPU=True, bboxes=bboxes_perm)
            xEmbedding = xEmbedding.numpy()
            for i in range(xEmbedding.shape[2]):
                for j in range(xEmbedding.shape[3]):
                    k = (image_base, perm, i, j)
                    xEmbedding_dict[k] = np.ascontiguousarray(xEmbedding[0, :, i, j])
                    prob_dict[k] = prob_map[i, j]

    return xEmbedding_dict, prob_dict

#returns pos_keys, neg_keys
def assign_labels(prob_dict):
    pos_keys = []
    neg_keys = []
    for k in sorted(prob_dict.keys()):
        prob = prob_dict[k]
        if prob == 1:
            pos_keys.append(k)
        elif prob == 0:
            neg_keys.append(k)
        else:
            r = random.uniform(0,1)
            if r < prob:
                pos_keys.append(k)
            else:
                neg_keys.append(k)

    return pos_keys, neg_keys

#image_filenames should be list of FULL paths to images
#they should have unique os.path.splitext(os.path.basename(image_filename))[0], i.e. image_base
#we will use (image_base, perm, i, j) as keys, where i and j are indices into embedding map and perm is the permutation of the image (always 0 for validation)
#bboxes_dict[image_base] should give a (possibly empty) list of bboxes, each one a dictionary with keys 'xmin', 'ymin', 'xmax', 'ymax'
#mode should be 'train' or 'val'
#params should be from relevance_config.py
#will yield X, y, where X is batch of embeddings and y is batch of 0s and 1s
#X is NCHW, where N is batch size and H=W=1
#y is (N,)
#if 'val' then we will yield None, None at the of validation epoch
#validation cannot guarantee a full-sized batch
def relevance_data_generator(image_filenames, bboxes_dict, mode, params):
    p = params
    assert(p.batch_size % 2 == 0)

    #keyed by (image_base, i, j, perm)
    xEmbedding_dict, prob_dict = run_embedder(image_filenames, bboxes_dict, mode, p)

    #now for the generator part!
    if mode == 'train':
        while True:
            pos_keys, neg_keys = assign_labels(prob_dict)
            k_pos = random.sample(pos_keys, p.batch_size // 2)
            k_neg = random.sample(neg_keys, p.batch_size // 2)
            X = []
            X.extend([xEmbedding_dict[k] for k in k_pos])
            X.extend([xEmbedding_dict[k] for k in k_neg])
            X = np.array(X, dtype='float32')[:,:,np.newaxis,np.newaxis]
            y = np.zeros(p.batch_size)
            y[:len(k_pos)] = 1.0
            y = y.astype('float32')
            yield X, y
    elif mode == 'val':
        while True:
            pos_keys, neg_keys = assign_labels(prob_dict)
            all_keys = pos_keys + neg_keys
            all_y = np.zeros(len(all_keys))
            all_y[:len(pos_keys)] = 1.0
            chunk_start = 0
            while chunk_start < len(all_keys):
                chunk_end = min(chunk_start + p.batch_size, len(all_keys))
                X = np.array([xEmbedding_dict[k] for k in all_keys[chunk_start:chunk_end]], dtype='float32')[:,:,np.newaxis,np.newaxis]
                y = copy.deepcopy(all_y[chunk_start:chunk_end]).astype('float32')
                yield X, y
                chunk_start += p.batch_size

            yield None, None
    else:
        assert(False)
