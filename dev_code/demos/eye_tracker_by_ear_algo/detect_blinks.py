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

csvData = [['mar', 'frames','original_mar']]
 
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
def mouth_aspect_ratio(mouth):
	A = dist.euclidean(mouth[3], mouth[9])
 	B = dist.euclidean(mouth[2], mouth[10])
	C = dist.euclidean(mouth[4], mouth[8])
	L = (A+B+C)/3.0
	D = dist.euclidean(mouth[0], mouth[6])
	mar=L/D
	return mar	
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-o", "--output", type = str, default = "outputy.mp4",
	help="path to output video file")
ap.add_argument("-f", "--fps", type=int, default=20,
	help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
	help="codec of output video")
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
total_frame_counter = 0
total_smile_counter = 0

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
vs = FileVideoStream(args["video"]).start()
fileStream = True
# vs = VideoStream(src=0).start()
# vs = VideoStream(args["video"]).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)
#
fourcc = cv2.VideoWriter_fourcc(*args["codec"])
writer = None
(h, w) = (None, None)
zeros = None
# loop over frames from the video stream
try:
	while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process

		if fileStream and not vs.more():
			break

		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		frame = vs.read()
		frame = imutils.resize(frame, width=450)
		#frame = imutils.rotate_bound(frame, 90)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		total_frame_counter += 1
		# img_name = "junk_images/opencv_frame_{}.png".format(total_frame_counter)
  		#cv2.imwrite(img_name, frame)

		# detect faces in the grayscale frame
		rects = detector(gray, 0)

		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
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

			#extract co-ordinates to the mouth and compute MAR
			mouth = shape[mStart:mEnd]

			#compute the mouth aspect ratio
			mar = mouth_aspect_ratio(mouth)

			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			#compute convex hull for the mouth, then visualize
			mouthHull = cv2.convexHull(mouth)
			cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
	  
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
				# reset the eye frame counter
				COUNTER = 0

			# if mar <= 0.3 or mar > 0.38 :
			# 	total_smile_counter +=1	
			if mar > 0.38:
				total_smile_counter +=1
				csvData.append([0.38,1,mar])
			elif mar > 0.3:
				total_smile_counter +=1
				csvData.append([0.3,1,mar])
			elif mar > 0.24:
				total_smile_counter +=1
				csvData.append([0.2,1,mar])
			elif mar > 0.1:
				csvData.append([0.1,1,mar])
			else:
				csvData.append([0,1,mar])			

			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "total_frames: {}".format(total_frame_counter),(10,60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# draw the Mar and total smile frames
			cv2.putText(frame, "MAR: {}".format(mar), (300, 60), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.putText(frame, "Smile_frames: {}".format(total_smile_counter), (10, 90), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	 	# write to frame
	 	if writer is None:
		 	(h, w) = frame.shape[:2]
			writer = cv2.VideoWriter(args["output"], fourcc, args["fps"],
				(w , h), True)
			zeros = np.zeros((h, w), dtype="uint8")	
		# show the frame
		cv2.imshow("Frame", frame)
		writer.write(frame)
		key = cv2.waitKey(1) & 0xFF

		#write it to a video file

	 
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	writer.release()
	cv2.destroyAllWindows()
	vs.stop()

# except Exception as e:
# 	pass
	# raise e
# else:
# 	pass
finally:
	counter_38 = 0
	counter_3 = 0
	counter_2 = 0
	counter_1 = 0
	counter_0 = 0
	for a,b,c in csvData:

		if a == 0.38:
			counter_38 +=1
		elif a == 0.3:
			counter_3 +=1
		elif a == 0.2:
			counter_2 +=1
		elif a == 0.1:
			counter_1 +=1
		else:
			counter_0 +=1
	csvData.append(["c38","c3","c2","c1","c0"])
	csvData.append([counter_38,counter_3,counter_2,counter_1,counter_0])
	csvData.append(["total_frames","total_smile_counter","Blinks_count"])
	csvData.append([total_frame_counter,total_smile_counter,TOTAL])

	with open('eye_blinks.csv', 'wb') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(csvData) 
