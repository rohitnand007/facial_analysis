# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from fps_calculator import fps_calculator
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import FileVideoStream
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import csv


 
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", required=True,
    help="path to input video ")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
frame_counter = 0
frames_in_sec, detected_frames , total_sec, blinks_in_sec, current_sec = 0,0,0,0,0
total_detected_frames = 0
cal_actual_sec = 0


#initialize the container for CSV file
csvData = [["in_sec","blinks_in_sec"]]

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = cv2.VideoCapture(args["video"]) 
#vs = FileVideoStream(args["video"]).start()
#time.sleep(1.0)
fps = fps_calculator(vs)
print("=======================================================:{}".format(fps))

# loop over frames from the video stream
try:
	while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
		(grabbed,frame) = vs.read()
		if grabbed:
			# grab the frame from the threaded video file stream, resize
			# it, and convert it to grayscale channels)

			frame = imutils.resize(frame, width=450)
			#frame = imutils.rotate_bound(frame, 90)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frame_counter += 1
			frames_in_sec += 1
			if (frames_in_sec - 1) == int(fps):
				total_sec += 1
				current_sec = 1
				frames_in_sec = 1
			print(frames_in_sec)
			#frame = imutils.rotate_bound(frame, 90)
			# img_name = "junk_images/opencv_frame_{}.png".format(total_frame_counter)
  			#cv2.imwrite(img_name, frame)
			# detect faces in the grayscale frame
			rects = detector(gray, 0)  	
			# loop over the face detections
			for rect in rects:
				# determine the facial landmarks for the face region, then
				# convert the facial landmark (x, y)-coordinates to a NumPy
				# array
				detected_frames += 1
				total_detected_frames += 1
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)

				# extract the left and right eye coordinates, then use the
				# coordinates to compute the eye aspect ratio for both eyes
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				leftEAR = eye_aspect_ratio(leftEye)
				rightEAR = eye_aspect_ratio(rightEye)

				# average the eye aspect ratio together for both eyes
				ear = (leftEAR + rightEAR) / 2.0	

				# compute the convex hull for the left and right eye, then
				# visualize each of the eyes
				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)	

				# check to see if the eye aspect ratio is below the blink
				# threshold, and if so, increment the blink frame counter
				if ear < EYE_AR_THRESH:
					COUNTER += 1
				# otherwise, the eye aspect ratio is not below the blink
				# threshold
				else:
					# if the eyes were closed for a sufficient number of
					# then increment the total number of blinks
					if COUNTER >= EYE_AR_CONSEC_FRAMES:
						TOTAL += 1
						blinks_in_sec += 1						
					# reset the eye frame counter
					COUNTER = 0
				
				# draw the total number of blinks on the frame along with
				# the computed eye aspect ratio for the frame
				cv2.putText(frame, "total_sec: {}".format(total_sec), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "frames_in_sec: {}".format(frames_in_sec),(10,60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
				cv2.putText(frame, "TOTAL: {}".format(TOTAL),(10,90),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
				cv2.putText(frame, "det_f: {}".format(detected_frames), (300, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "t_d: {}".format(total_detected_frames),(300,60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
				cv2.putText(frame, "b/s: {}".format(blinks_in_sec), (300, 90),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			if current_sec == 1 and detected_frames >= int(0.6 * frames_in_sec):
				cal_actual_sec = int(frame_counter/fps)
				csvData.append([cal_actual_sec,blinks_in_sec])
				blinks_in_sec = 0
				current_sec = 0	
				detected_frames = 1
			elif current_sec ==1 and detected_frames < int(0.6 * frames_in_sec):
				cal_actual_sec = int(frame_counter/fps)
				csvData.append([cal_actual_sec, -1])
				blinks_in_sec = 0
				current_sec = 0	
				detected_frames = 1					
		else:
			break
	# do a bit of cleanup
	cv2.destroyAllWindows()	

except Exception as e:
	raise e
else:
	pass
finally:
	csvData.append(["total_sec","total_blinks", "total_frames"])
	csvData.append([cal_actual_sec,TOTAL, frame_counter])
	file_name = "output_csv/" + args["video"].split("/")[-1].split(".")[0] + ".csv"

	with open(file_name, 'wb') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(csvData) 



