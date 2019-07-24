# fps = vs.get(cv2.CAP_PROP_FPS)
# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from imutils.video.count_frames import *
from imutils.video import FPS
import cv2


def fps_calculator(video):
	video = cv2.VideoCapture(video)
	fps = video.get(cv2.CAP_PROP_FPS)
	override = False
	if fps > 100:
		override = True
		frame_count = count_frames_manual(video)
		video.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
		time_elaspsed = video.get(cv2.CAP_PROP_POS_MSEC)/1000
		fps = frame_count/float(time_elaspsed)
	return fps	

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", required=True,
#     help="path to input video ")
# ap.add_argument("-o", "--override", type=int, default=-1,
# 	help="whether to force manual frame count")
# args = vars(ap.parse_args())

# # start the video stream thread
# print("[INFO] starting video stream thread...")
# fps1 = FPS().start()
# vs = cv2.VideoCapture(args["video"])
# fileStream = True
# vs.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
# while True:
# 	(grabbed,frame) = vs.read()
# 	if grabbed:	
# 		cv2.imshow("Frame", frame)
# 		key = cv2.waitKey(1) & 0xFF
# 		fps1.update()
# 	else:
# 		break	

# fps = vs.get(cv2.CAP_PROP_FPS)
# override = False if args["override"] < 0 else True
# total = count_frames_manual(vs)#count_frames(args["video"], override=override)
# print("..........................................{}".format(fps))
# fps1.stop()
# elapsed = fps1.elapsed()
# time_elaspsed = vs.get(cv2.CAP_PROP_POS_MSEC)

# file1 = open("results.txt","a")
# str2 = "Frame rate for this video [{}] : Fps = [{}] & total_frames: [{}]/n.....".format(args["video"],fps,total)
# file1.write(str2)
# file1.close()
# print(str2) 
# print("Time elapsed is....:{}".format(elapsed))
# print("Time elaspsed from method is...................:{}".format(time_elaspsed))