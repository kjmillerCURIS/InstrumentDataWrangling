import os
import sys

class VideoWranglingParams:
    def __init__(self):

        #this section is used by download_videos.py ONLY
        #dictionary that maps from video category to list of URLs
        #each video filename will be something like <CATEGORY>-<HASH_OF_URL>-<ITAG>.mp4
        #the data-wrangler will check for collisions in the hashes
        #we'll make a pickle dict that maps hash back to URL
        self.pytube_kwargs = {'file_extension' : 'mp4', 'adaptive' : True, 'mime_type' : 'video/mp4'}
        #720p seems like a good compromise for now
        self.resolutions = ['720p', '480p', '360p', '240p', '144p']
        self.video_urls = {'ortho_and_general' : [], 'how_its_made_instruments' : [], 'sterilizing_instruments' : []}
        self.video_urls['ortho_and_general'].append('https://www.youtube.com/watch?v=vTnuyCHunOk')
        self.video_urls['ortho_and_general'].append('https://www.youtube.com/watch?v=dIf3U9NTsQs')
        self.video_urls['ortho_and_general'].append('https://www.youtube.com/watch?v=GEgRaoNHKA4') #this one is vertical :(
        self.video_urls['ortho_and_general'].append('https://www.youtube.com/watch?v=QUc7r241rrk')
        self.video_urls['ortho_and_general'].append('https://www.youtube.com/watch?v=0F9NwKWYack')
        self.video_urls['ortho_and_general'].append('https://www.youtube.com/watch?v=RGa4YsCUg_E')
        self.video_urls['ortho_and_general'].append('https://www.youtube.com/watch?v=RDR7T_0vB8E')
        self.video_urls['ortho_and_general'].append('https://www.youtube.com/watch?v=XDTWRMs07XU')
        self.video_urls['how_its_made_instruments'].append('https://www.youtube.com/watch?v=Xw7s4iNsAfM')
        self.video_urls['how_its_made_instruments'].append('https://www.youtube.com/watch?v=3bQTlru_0dQ')
        self.video_urls['how_its_made_instruments'].append('https://www.youtube.com/watch?v=inZczv3bLu4')
        self.video_urls['how_its_made_instruments'].append('https://www.youtube.com/watch?v=GIKTG_g3z6o')
        self.video_urls['sterilizing_instruments'].append('https://www.youtube.com/watch?v=p_52PvJVxG8')
        self.video_urls['sterilizing_instruments'].append('https://www.youtube.com/watch?v=VGLJO6MhtwE')
        #TODO: Consider adding more URLs
        
        #this section is used by extract_frames.py ONLY
        #each image filename will be something like <VIDEO_BASENAME>-frame#####.jpg
        #where the frame number is the actual frame number, not just 1,2,3,...,1000
        self.num_frames_per_video = 1000 #keep any one video from having too much influence
        self.frame_sampling_mode = 'random.sample' #frames_to_use = random.sample(all_frames, 1000)
        self.frame_sampling_seed = 0 #only call random.seed() ONCE - this is just for reproducibility
        self.side_cutoff_dict = {} #will map <VIDEO_BASENAME> to number of pixels to cut off of both sides, ONLY including videos that are vertical. #TODO: FILL THIS IN!!!

def grab_params(params_key):
    return eval(params_key + '()')
