import os
import sys
import hashlib
import pickle
from tqdm import tqdm
from pytube import YouTube
from video_wrangling_config import grab_params

def download_videos(params_key, video_dir, hash_dict_filename):
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    p = grab_params(params_key)
    thingies = [] #(category, URL, hash(URL)) triplets
    categories = sorted(p.video_urls.keys())
    for category in categories:
        for video_url in p.video_urls[category]:
            video_hash = hashlib.md5(video_url.encode('utf8')).hexdigest()
            thingies.append((category, video_url, video_hash))

    #check for collisions
    all_hashes = [x[2] for x in thingies]
    assert(len(set(all_hashes)) == len(all_hashes))

    #make hash dict
    hash_dict = {x[2] : x[1] for x in thingies}
    with open(hash_dict_filename, 'wb') as f:
        pickle.dump(hash_dict, f)

    for category, video_url, video_hash in tqdm(thingies):
        yt = YouTube(video_url)
        best_stream = None
        for res in p.resolutions:
            my_streams = yt.streams.filter(**(p.pytube_kwargs), res=res)
            if len(my_streams) > 0:
                best_stream = my_streams[0] #arbitrarily pick first one
                break

        if best_stream is None:
            print('Could not download from URL "%s" in category "%s"'%(video_url, category))
            continue

        filename = category + '-' + video_hash + '-' + str(best_stream.itag)
        best_stream.download(output_path=video_dir, filename=filename)

def usage():
    print('Usage: python download_videos.py <params_key> <video_dir> <hash_dict_filename>')

if __name__ == '__main__':
    download_videos(*(sys.argv[1:]))
