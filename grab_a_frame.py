import os
import sys
import cv2

def grab_a_frame(video_filename, k, image_filename):
    k = int(k)
    cap = cv2.VideoCapture(video_filename)
    for i in range(k+1):
        ret, numI = cap.read()
        assert(ret)

    cv2.imwrite(image_filename, numI)

def usage():
    print('Usage: python grab_a_frame.py <video_filename> <k> <image_filename>')

if __name__ == '__main__':
    grab_a_frame(*(sys.argv[1:]))
