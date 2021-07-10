import os
import sys
import glob
import random
import shutil
from tqdm import tqdm

def pick_random_sample(src_dir, k, random_seed, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    k = int(k)
    random_seed = int(random_seed)
    random.seed(random_seed)
    thingies = sorted(glob.glob(os.path.join(src_dir, '*')))
    thingies = [thingy for thingy in thingies if not os.path.isdir(thingy)]
    thingies = random.sample(thingies, k)
    for thingy in tqdm(thingies):
        shutil.copy(thingy, dst_dir)

def usage():
    print('Usage: python pick_random_sample.py <src_dir> <k> <random_seed> <dst_dir>')

if __name__ == '__main__':
    pick_random_sample(*(sys.argv[1:]))
