# fps = vs.get(cv2.CAP_PROP_FPS)
# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="path to input video ")
args = vars(ap.parse_args())

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = cv2.VideoCapture(args["video"])
fileStream = True
# vs = VideoStream(src=0).start()
# vs = VideoStream(args["video"]).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)
fps = vs.get(cv2.CAP_PROP_FPS)

file1 = open("results.txt","a")
str2 = "Frame rate for this video [{}] : [{}]/n.....".format(args["video"],fps)
file1.write(str2)
file1.close()
print(str2) 