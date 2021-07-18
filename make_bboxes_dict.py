import os
import sys
import glob
import json
import pickle

#incoming bbox is list of lists of floats
def convert_bbox(in_bbox):
    xmin = int(round(min(in_bbox[0][0], in_bbox[1][0])))
    xmax = int(round(max(in_bbox[0][0], in_bbox[1][0])))
    ymin = int(round(min(in_bbox[0][1], in_bbox[1][1])))
    ymax = int(round(max(in_bbox[0][1], in_bbox[1][1])))
    return {'xmin' : xmin, 'ymin' : ymin, 'xmax' : xmax, 'ymax' : ymax}

def make_bboxes_dict(image_dir, json_dir, bboxes_dict_filename):
    images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    bboxes_dict = {}
    num_empty = 0
    num_empty_at_end = 0
    for image in images:
        k = os.path.splitext(os.path.basename(image))[0]
        json_filename = os.path.join(json_dir, k + '.json')
        if os.path.exists(json_filename):
            num_empty_at_end = 0
            with open(json_filename, 'r') as f:
                d = json.load(f)

            bboxes = []
            for in_bbox in d['shapes']:
                bbox = convert_bbox(in_bbox['points'])
                bboxes.append(bbox)

            bboxes_dict[k] = bboxes
                
        else:
            num_empty += 1
            num_empty_at_end += 1
            bboxes_dict[k] = []

    print('empties = %d/%d'%(num_empty, len(images)))
    print('empties-at-end = %d/%d'%(num_empty_at_end, len(images)))
    with open(bboxes_dict_filename, 'wb') as f:
        pickle.dump(bboxes_dict, f)

def usage():
    print('Usage: python make_bboxes_dict.py <image_dir> <json_dir> <bboxes_dict_filename>')

if __name__ == '__main__':
    make_bboxes_dict(*(sys.argv[1:]))
