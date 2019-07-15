# fps = vs.get(cv2.CAP_PROP_FPS)
# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from imutils.video import count_frames
import argparse
import time
import dlib
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="path to input video ")
ap.add_argument("-o", "--override", type=int, default=-1,
	help="whether to force manual frame count")
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
override = False if args["override"] < 0 else True
total = count_frames(args["video"], override=override)

file1 = open("results.txt","a")
str2 = "Frame rate for this video [{}] : Fps = [{}] & total_frames: [{}]/n.....".format(args["video"],fps,total)
file1.write(str2)
file1.close()
print(str2) 