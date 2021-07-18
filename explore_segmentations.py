import os
import sys
import copy
import cv2
import glob
import numpy as np
import pickle
import random
import torch
from torch import nn
from relevance_config import grab_params
from relevance_embedder import RelevanceEmbedder
from relevance_model import RelevanceModel

DISPLAY_WIDTH = 1800
DISPLAY_HEIGHT = 900
DISPLAY_NUM_ROWS = 3
assert(DISPLAY_HEIGHT % DISPLAY_NUM_ROWS == 0)
DISPLAY_IMAGE_HEIGHT = DISPLAY_HEIGHT // DISPLAY_NUM_ROWS
MAX_NUM_EXAMPLES = 20
FOREGROUND_MASK_COLOR = (0,255,0)
BACKGROUND_MASK_COLOR = (255,0,255)

RESULT_BASES = [\
'RelevanceParams16Comp1Trust',\
'RelevanceParams1Comp1Trust',\
'RelevanceParams32Comp1Trust',\
'RelevanceParams16CompHalfTrust',\
'RelevanceParams1CompHalfTrust',\
'RelevanceParams32CompHalfTrust',\
'RelevanceParams16CompInfTrust',\
'RelevanceParams1CompInfTrust',\
'RelevanceParams32CompInfTrust'\
]

RESULT_DIR = '/home/kevin/InstrumentData/relevance_results'
RESULT_FILENAMES = [os.path.join(RESULT_DIR, result_base + '-result.pkl') for result_base in RESULT_BASES]

#all keys can be uppercase or lowercase
#'+'/'-' will cycle through models
#'r' means resample testers
#'f' means toggle foreground mask
#'b' means toggle background mask
#'q' means quit

#numI should be original image, i.e. from cv2.imread()
def make_display_image(numI):
    new_width = int(round(numI.shape[1] * DISPLAY_IMAGE_HEIGHT / numI.shape[0]))
    return cv2.resize(numI, (new_width, DISPLAY_IMAGE_HEIGHT))

#mask should be what comes out of model, (H,W) shape, uint8, 0 and 1
#numIdisplay is display image, only using for shape
#returns display_mask, same width and height as numIdisplay, uint8, 0 and 1
def make_display_mask(mask, numIdisplay):
    assert(len(mask.shape) == 2)
    assert(mask.dtype == 'uint8')
    assert(np.all((mask == 0) | (mask == 1)))

    #try to upsample by np.repeat()
    src_diag = np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)
    dst_diag = np.sqrt(numIdisplay.shape[0]**2 + numIdisplay.shape[1]**2)
    num_repeats = int(round(dst_diag / src_diag))
    display_mask = np.repeat(np.repeat(mask, num_repeats, axis=0), num_repeats, axis=1)

    #resize to fit numIdisplay exactly
    return cv2.resize(display_mask, (numIdisplay.shape[1], numIdisplay.shape[0]))

#image_filename_dict maps each image_base to its full filename
#this will return:
#* example_image_bases ==> list of image bases that we'll display
#* numI_dict ==> map each image_base to its numI (which we'll feed into inference)
#* numIdisplay_dict ==> map each image_base to its numIdisplay
#* origin_dict ==> map each image_base to the (x,y) of where to put its upper-left corner in the collage
#It will be someone else's job to do inference and make the masks
def resample_images(image_filename_dict):
    all_image_bases = sorted(image_filename_dict.keys())
    cur_origin = (0,0)
    super_image_bases = random.sample(all_image_bases, MAX_NUM_EXAMPLES)
    example_image_bases = []
    numI_dict = {}
    numIdisplay_dict = {}
    origin_dict = {}
    for image_base in super_image_bases:
        numI = cv2.imread(image_filename_dict[image_base])
        numIdisplay = make_display_image(numI)
        should_quit = False
        while True:
            #if we CANNOT fit our height
            if cur_origin[1] + DISPLAY_IMAGE_HEIGHT > DISPLAY_HEIGHT:
                should_quit = True
                break

            #if we CAN fit our width (and our height, as implied by the thingy above)
            if cur_origin[0] + numIdisplay.shape[1] <= DISPLAY_WIDTH:
                break

            #go to the next row (because we CANNOT fit our width)
            cur_origin = (0, cur_origin[1] + DISPLAY_IMAGE_HEIGHT)

        if should_quit:
            break

        example_image_bases.append(image_base)
        numI_dict[image_base] = numI
        numIdisplay_dict[image_base] = numIdisplay
        origin_dict[image_base] = cur_origin

        #increment cur_origin
        cur_origin = (cur_origin[0] + numIdisplay.shape[1], cur_origin[1])

    return example_image_bases, numI_dict, numIdisplay_dict, origin_dict

#numI is original image
#returns mask, which can be passed into mask_display_mask
def do_inference(numI, my_embedder, my_model):
    my_model.eval()
    with torch.no_grad():
        xEmbedding = my_embedder(numI)
        logits = my_model(xEmbedding).to('cpu').numpy()[0,0,:,:]

    mask = (logits > 0).astype('uint8')
    return mask

#display_mask_dict is keyed by result_filename (yes, the full path) and then image_base
#if an entry is missing, we'll populate it
#return the populated display_mask_dict
def load_new_result(result_filename, example_image_bases, numI_dict, numIdisplay_dict, display_mask_dict):
    if result_filename not in display_mask_dict:
        display_mask_dict[result_filename] = {}

    everything_already_there = True
    for image_base in example_image_bases:
        if image_base not in display_mask_dict[result_filename]:
            everything_already_there = False
            break

    if everything_already_there:
        return display_mask_dict

    #load embedder and model
    with open(result_filename, 'rb') as f:
        result = pickle.load(f)

    p = grab_params(result['params_key'])
    my_embedder = RelevanceEmbedder(p)
    my_model = RelevanceModel(p)
    my_model.load_state_dict(torch.load(result['model_filename']))
    my_model.eval()

    for image_base in example_image_bases:
        if image_base in display_mask_dict[result_filename]:
            continue

        mask = do_inference(numI_dict[image_base], my_embedder, my_model)
        display_mask = make_display_mask(mask, numIdisplay_dict[image_base])
        display_mask_dict[result_filename][image_base] = display_mask

    return display_mask_dict

#returns collage
def make_collage(example_image_bases, numIdisplay_dict, origin_dict, display_mask_dict, foreground_flag, background_flag):
    numIcollage = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype='uint8')
    for image_base in example_image_bases:
        numIvis = copy.deepcopy(numIdisplay_dict[image_base])
        display_mask = display_mask_dict[image_base]
        if foreground_flag:
            numIvis[display_mask > 0,:] = FOREGROUND_MASK_COLOR
        if background_flag:
            numIvis[display_mask == 0,:] = BACKGROUND_MASK_COLOR
        origin = origin_dict[image_base]
        numIcollage[origin[1]:origin[1]+numIvis.shape[0], origin[0]:origin[0]+numIvis.shape[1], :] = numIvis[:,:,:]

    return numIcollage

def explore_segmentations(image_dir):
    cv2.namedWindow('meow')
    cur_result_index = 0
    foreground_flag = 0 #show original pixels for foreground
    background_flag = 1 #make the background purple
    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    image_filename_dict = {os.path.splitext(os.path.basename(image_filename))[0] : image_filename for image_filename in image_filenames}
    example_image_bases, numI_dict, numIdisplay_dict, origin_dict = resample_images(image_filename_dict)
    display_mask_dict = {}
    print(RESULT_FILENAMES[cur_result_index])
    display_mask_dict = load_new_result(RESULT_FILENAMES[cur_result_index], example_image_bases, numI_dict, numIdisplay_dict, display_mask_dict)
    while True:
        numIcollage = make_collage(example_image_bases, numIdisplay_dict, origin_dict, display_mask_dict[RESULT_FILENAMES[cur_result_index]], foreground_flag, background_flag)
        cv2.imshow('meow', numIcollage)
        while True:
            k = cv2.waitKey(0)
            if k in [ord('f'), ord('F')]:
                foreground_flag = 1 - foreground_flag
                break
            elif k in [ord('b'), ord('B')]:
                background_flag = 1 - background_flag
                break
            elif k in [ord('r'), ord('R')]:
                example_image_bases, numI_dict, numIdisplay_dict, origin_dict = resample_images(image_filename_dict)
                display_mask_dict = {}
                display_mask_dict = load_new_result(RESULT_FILENAMES[cur_result_index], example_image_bases, numI_dict, numIdisplay_dict, display_mask_dict)
                break
            elif k in [ord('-'), ord('_')]:
                cur_result_index = (cur_result_index - 1) % len(RESULT_FILENAMES)
                print(RESULT_FILENAMES[cur_result_index])
                display_mask_dict = {}
                display_mask_dict = load_new_result(RESULT_FILENAMES[cur_result_index], example_image_bases, numI_dict, numIdisplay_dict, display_mask_dict)
                break
            elif k in [ord('='), ord('+')]:
                cur_result_index = (cur_result_index - 1) % len(RESULT_FILENAMES)
                print(RESULT_FILENAMES[cur_result_index])
                display_mask_dict = {}
                display_mask_dict = load_new_result(RESULT_FILENAMES[cur_result_index], example_image_bases, numI_dict, numIdisplay_dict, display_mask_dict)
                break
            elif k in [ord('q'), ord('Q')]:
                return

def usage():
    print('Usage: explore_segmentations.py <image_dir>')

if __name__ == '__main__':
    explore_segmentations(*(sys.argv[1:]))
