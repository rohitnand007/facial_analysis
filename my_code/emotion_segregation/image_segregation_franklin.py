# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from helper_methods import *
from scipy.spatial import distance as dist
from sklearn.cluster import MeanShift
import math 
from imutils import face_utils
from imutils.face_utils import FaceAligner
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import csv
import os
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default=None,
    help="path to input video ")
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to dlib face detector")
args = vars(ap.parse_args())

img_vector_data = {'vectorised_data':[]}

COUNTER = 0  
detected_counter = []
undetected_counter = [] 
alined_undetected_counter = []          
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])
fa = FaceAligner(predictor, desiredFaceWidth=256)

print(detector)

# start the video stream thread
print("[INFO] starting video stream thread...")
video = args["video"]
video_title_path = os.getcwd() + "/data_folder" + "/" +  video.split("/")[-1].split(".")[0]
vs = cv2.VideoCapture(video) 

#create the parent directory for image storage:
ini_img_path = create_parent_dir(video_title_path)
print("===========================================.parent_dir created:{}".format(ini_img_path))

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
            #increase the frame counter by 1
            COUNTER += 1

            # frame = imutils.rotate_bound(frame, 270)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale frame
            rects = detector(gray, 0)  
            if len(rects) > 0:
                for (i,rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                    print("+++++++++++++++++++++++++++>{}".format(COUNTER))
                    shape = predictor(gray, rect)
                    faceAligned = fa.align(frame, gray, rect)
                    grayAlined = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
                    alinedRect = detector(grayAlined,0)
                    if len(alinedRect) > 0:
                        cv2.imwrite(ini_img_path + "/detected_images/" + str(COUNTER) + '.jpg', faceAligned)
                        rectAlined = alinedRect[0]
                        shape = predictor(grayAlined,rectAlined)
                        shape = face_utils.shape_to_np(shape)
                        data = get_dist_angle(shape)
                        img_vector_data['vectorised_data'].append(data) 
                        detected_counter.append(COUNTER) 
                        # (x, y, w, h) = face_utils.rect_to_bb(rectAlined)
                        # cv2.rectangle(faceAligned, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        alined_undetected_counter.append(COUNTER) 
                        cv2.imwrite(ini_img_path + "/alined_undetected_images/" + str(COUNTER) + '.jpg', faceAligned)
                        print("alined_undetected********************************************")
                          
            else: 
                undetected_counter.append(COUNTER) 
                cv2.imwrite(ini_img_path + "/undetected_images/" + str(COUNTER) + '.jpg', frame)
                print("undetected********************************************")

        
            # cv2.imshow("Frame", faceAligned)
            # key = cv2.waitKey(1) & 0xFF         
        else:
            break
    # do a bit of cleanup
    cv2.destroyAllWindows()

except Exception as e:
    pass
    raise e
else:
    pass
finally:
    csvData = []
    csvTitleBar = []
    for i in range(1,69):
        csvTitleBar.append("dist_pt_"+str(i))
        csvTitleBar.append("angle_pt_"+str(i))
    csvData.append(csvTitleBar)
    for d in img_vector_data['vectorised_data']:
        csvData.append(d)
    file_name1 = video_title_path+"/"+ video.split("/")[-1].split(".")[0] + "_detected.csv"

    with open(file_name1, 'wb') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData) 

#csvfile for mapping and undetected data
    csvData2 = []
    csvData2.append(["detected_row_number","corresponding_image_filename"])
    file_name2 = video_title_path+"/"+ video.split("/")[-1].split(".")[0] + "_mapping_and_undetected.csv"

    for (i,img_num) in enumerate(detected_counter, start=2):
        csvData2.append([i, str(img_num)+".jpg"])

    csvData2.append(["serial_no","undetected_image_filename"])
    for (i,undetected) in enumerate(undetected_counter, start=1):
        csvData2.append([i, str(undetected)+".jpg"])

    csvData2.append(["Total Detected Frames","Total undetected Frames", "Total_alined_undetected"]) 
    csvData2.append([len(detected_counter),len(undetected_counter), len(alined_undetected_counter)])   

    with open(file_name2, 'wb') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData2)





