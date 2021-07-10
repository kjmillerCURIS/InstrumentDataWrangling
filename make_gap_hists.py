import os
import sys
import glob
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def make_gap_hists(frame_dir, hist_dir):
    if not os.path.exists(hist_dir):
        os.makedirs(hist_dir)

    images = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))
    indices_dict = {}
    for image in tqdm(images):
        video_base, t = os.path.splitext(os.path.basename(image))[0].split('-frame')
        t = int(t)
        if video_base not in indices_dict:
            indices_dict[video_base] = []

        indices_dict[video_base].append(t)

    for video_base in sorted(indices_dict.keys()):
        indices = np.array(sorted(indices_dict[video_base]))
        gaps = indices[1:] - indices[:-1]
        plt.clf()
        plt.hist(gaps)
        plt.xlabel('gap (frames)')
        plt.ylabel('freq')
        plt.savefig(os.path.join(hist_dir, video_base + '.png'))
        plt.clf()

def usage():
    print('Usage: python make_gap_hists.py <frame_dir> <hist_dir>')

if __name__ == '__main__':
    make_gap_hists(*(sys.argv[1:]))
