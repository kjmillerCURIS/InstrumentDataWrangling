import os
import sys
import cv2
import glob
import json
import numpy as np
import pickle
import random
import torch
from torch import nn
from relevance_config import grab_params
from relevance_data_generator import check_inputs, relevance_data_generator
from relevance_model import RelevanceModel

#does a training step and returns the loss of the one training batch
def train_step(X, y, my_model, my_opt):
    my_model.train()
    X = torch.tensor(X).to('cuda')
    y = torch.tensor(y).to('cuda')
    logits = my_model(X)
    my_loss = nn.BCEWithLogitsLoss()(torch.flatten(logits), y)
    my_opt.zero_grad()
    my_loss.backward()
    my_opt.step()
    return my_loss.item() #item() should return a float, which is automatically on CPU and is impossible to attach to the computation graph (duh!)

#gt should be 0 or 1
def one_BCE(logit, gt):
    if gt == 0:
        #-log(1-sigmoid(z)) = -log(exp(-z)) + log(1+exp(-z)) = z+log(1+exp(-z)) = z+log(1+exp(z))-log(exp(z))
        #= log(1+exp(z))
        return np.log(1 + np.exp(logit))
    elif gt == 1:
        #-log(sigmoid(z)) = -log(1) + log(1+exp(-z))
        #= log(1+exp(-z))
        return np.log(1 + np.exp(-logit))
    else:
        assert(False)

#returns dictionary with keys {'gt_pos_N', 'gt_pos_loss', 'gt_neg_N', 'gt_neg_loss', 'balanced_loss'}
def val_step(val_genny, my_model):
    my_model.eval()
    gt_pos_sum_BCE = 0.0
    gt_pos_N = 0
    gt_neg_sum_BCE = 0.0
    gt_neg_N = 0
    for X, y in val_genny:
        if X is None:
            break

        with torch.no_grad():
            X = torch.tensor(X).to('cuda')
            gts = y
            y = torch.tensor(y).to('cuda')
            logits = my_model(X)
            logits = torch.flatten(logits).to('cpu').numpy()
            assert(logits.shape == gts.shape)
            for logit, gt in zip(logits, gts):
                one_loss = one_BCE(logit, gt)
                if gt == 0:
                    gt_neg_sum_BCE += one_loss
                    gt_neg_N += 1
                elif gt == 1:
                    gt_pos_sum_BCE += one_loss
                    gt_pos_N += 1
                else:
                    assert(False)

    gt_pos_loss = gt_pos_sum_BCE / gt_pos_N
    gt_neg_loss = gt_neg_sum_BCE / gt_neg_N
    balanced_loss = 0.5 * gt_pos_loss + 0.5 * gt_neg_loss
    return {'gt_pos_N' : gt_pos_N, 'gt_neg_N' : gt_neg_N, 'gt_pos_loss' : gt_pos_loss, 'gt_neg_loss' : gt_neg_loss, 'balanced_loss' : balanced_loss}

#need to take in result because it will already have stuff in it
#will save best model whenever it improves, and save result to result_filename at end
#also save result to result_filename at every validation step, in case we want to peek at it :)
def training_loop(train_genny, val_genny, model_filename, result, result_filename, params):
    p = params
    result['train_losses'] = {}
    result['val_losses'] = {}
    result['is_done'] = False
    result['model_filename'] = os.path.abspath(model_filename)
    best_val_loss = float('+inf')
    my_model = RelevanceModel(p)
    my_opt = torch.optim.Adam(my_model.parameters(), lr=p.learning_rate)
    for t in range(p.num_training_steps):
        if t % p.val_freq == 0:
            val_result = val_step(val_genny, my_model)
            result['val_losses'][t] = val_result
            val_loss = val_result['balanced_loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                #no need to deepcopy the my_model.state_dict(), because saving it would make a copy on disk
                #HOWEVER, if you were to hold on to my_model.state_dict() it in RAM, then you WOULD need to deepcopy it!
                torch.save(my_model.state_dict(), model_filename)

                print('\n*t=%d/%d, val_loss=%f\n'%(t, p.num_training_steps, val_loss))
            else:
                print('\nt=%d/%d, val_loss=%f\n'%(t, p.num_training_steps, val_loss))
            
            with open(result_filename, 'wb') as f:
                pickle.dump(result, f)

        X, y = next(train_genny)
        train_loss = train_step(X, y, my_model, my_opt)
        print('.', end='', flush=True)
        result['train_losses'][t] = train_loss

    result['is_done'] = True
    with open(result_filename, 'wb') as f:
        pickle.dump(result, f)

def make_train_val_split_helper(image_bases, num_val):
    val_image_bases = random.sample(image_bases, num_val)
    train_image_bases = [image_base for image_base in image_bases if image_base not in val_image_bases]
    return train_image_bases, val_image_bases

#return train_image_filenames, val_image_filenames
def make_train_val_split(image_filenames, bboxes_dict, params):
    p = params
    check_inputs(image_filenames, bboxes_dict)
    image_bases_to_filenames = {os.path.splitext(os.path.basename(image_filename))[0] : image_filename for image_filename in image_filenames}
    image_bases = sorted(image_bases_to_filenames.keys())
    empty_image_bases = []
    full_image_bases = []
    for image_base in image_bases:
        if len(bboxes_dict[image_base]) == 0:
            empty_image_bases.append(image_base)
        else:
            full_image_bases.append(image_base)

    train_image_bases = []
    val_image_bases = []
    num_val = int(round(p.val_prop * len(image_bases)))
    num_val_empty = int(round(p.prop_no_bbox_in_val * p.val_prop * len(image_bases)))
    num_val_full = num_val - num_val_empty
    t_full, v_full = make_train_val_split_helper(full_image_bases, num_val_full)
    train_image_bases.extend(t_full)
    val_image_bases.extend(v_full)
    t_empty, v_empty = make_train_val_split_helper(empty_image_bases, num_val_empty)
    train_image_bases.extend(t_empty)
    val_image_bases.extend(v_empty)
    train_image_filenames = [image_bases_to_filenames[image_base] for image_base in train_image_bases]
    val_image_filenames = [image_bases_to_filenames[image_base] for image_base in val_image_bases]
    return train_image_filenames, val_image_filenames

#bboxes_dict_filename will be a pickle
#some other script can process the labelMe output
#will return train_genny, val_genny, result
#take in result because someone else creates it
def get_generators(image_dir, bboxes_dict_filename, result, params):
    p = params
    result['bboxes_dict_filename'] = os.path.abspath(bboxes_dict_filename)
    result['image_dir'] = os.path.abspath(image_dir)
    with open(bboxes_dict_filename, 'rb') as f:
        bboxes_dict = pickle.load(f)

    result['bboxes_dict'] = bboxes_dict
    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    train_image_filenames, val_image_filenames = make_train_val_split(image_filenames, bboxes_dict, p)
    result['train_image_filenames'] = train_image_filenames
    result['val_image_filenames'] = val_image_filenames
    train_genny = relevance_data_generator(train_image_filenames, bboxes_dict, 'train', p)
    val_genny = relevance_data_generator(val_image_filenames, bboxes_dict, 'val', p)
    return train_genny, val_genny, result

def relevance_training(image_dir, bboxes_dict_filename, params_key, model_filename, result_filename):
    if not os.path.exists(os.path.dirname(model_filename)):
        os.makedirs(os.path.dirname(model_filename))

    if not os.path.exists(os.path.dirname(result_filename)):
        os.makedirs(os.path.dirname(result_filename))

    p = grab_params(params_key)
    result = {'params_key' : params_key, 'params' : p.__dict__}
    train_genny, val_genny, result = get_generators(image_dir, bboxes_dict_filename, result, p)
    training_loop(train_genny, val_genny, model_filename, result, result_filename, p)

def usage():
    print('Usage: python relevance_training.py <image_dir> <bboxes_dict_filename> <params_key> <model_filename> <result_filename>')

if __name__ == '__main__':
    relevance_training(*(sys.argv[1:]))
