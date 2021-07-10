import os
import sys
import cv2
import glob
import numpy as np
import random
from tqdm import tqdm
from video_wrangling_config import grab_params

def extract_frames_one_video(video, params, frame_dir):
    p = params
    cap = cv2.VideoCapture(video)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    assert(num_frames - int(num_frames) == 0.0)
    num_frames = int(num_frames)
    if num_frames < p.num_frames_per_video:
        print('CAUTION: video "%s" only had %d frames, less than the usual %d-frame sample'%(video, num_frames, p.num_frames_per_video))

    video_base = os.path.splitext(os.path.basename(video))[0]
    cutoff = None
    if video_base in p.side_cutoff_dict:
        cutoff = p.side_cutoff_dict[video_base]
        assert(cutoff > 0)

    assert(p.frame_sampling_mode == 'random.sample')
    frame_indices = set(random.sample(range(num_frames), min(p.num_frames_per_video, num_frames)))
    for t in tqdm(range(num_frames)):
        ret, numI = cap.read()
        assert(ret)
        if t not in frame_indices:
            continue

        if cutoff is not None:
            assert(numI.shape[1] > 2 * cutoff)
            numI = np.ascontiguousarray(numI[:,cutoff:-cutoff,:])

        frame_filename = os.path.join(frame_dir, video_base + '-frame%09d.jpg'%(t))
        cv2.imwrite(frame_filename, numI)

def extract_frames(video_dir, params_key, frame_dir):
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    p = grab_params(params_key)

    #set random seed ONCE
    random.seed(p.frame_sampling_seed)

    videos = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
    for video in tqdm(videos):
        extract_frames_one_video(video, p, frame_dir)

def usage():
    print('Usage: python extract_frames.py <video_dir> <params_key> <frame_dir>')

if __name__ == '__main__':
    extract_frames(*(sys.argv[1:]))
