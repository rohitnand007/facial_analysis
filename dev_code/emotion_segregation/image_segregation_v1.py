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
args = vars(ap.parse_args())

img_vector_data = {'vectorised_data':[]}

COUNTER = 0  
measure_x = -1
measure_y = -1
detected_counter = []
undetected_counter = [] 
alined_undetected_counter = [] 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

print(detector)

# start the video stream thread
print("[INFO] starting video stream thread...")
video = args["video"]
video_title_path = os.getcwd() + "/" +  video.split("/")[-1].split(".")[0]
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
                    # shape = predictor(gray, rect)
                    faceAligned = fa.align(frame, gray, rect)
                    grayAlined = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
                    alinedRect = detector(grayAlined,0)
                    if len(alinedRect) > 0:
                        cv2.imwrite(ini_img_path + "/detected_images/" + str(COUNTER) + '.jpg', faceAligned)
                        rectAlined = alinedRect[0]
                        shape = predictor(grayAlined,rectAlined)
                        shape = face_utils.shape_to_np(shape)
                        if measure_x < 0 and measure_y < 0:
                            measure_x = get_measure_x(shape)
                            measure_y = get_measure_y(shape)
                        data = get_dist_angle(shape, measure_x, measure_y)
                        img_vector_data['vectorised_data'].append(data) 
                        detected_counter.append(COUNTER) 
                        # (x, y, w, h) = face_utils.rect_to_bb(rectAlined)
                        # cv2.rectangle(faceAligned, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        alined_undetected_counter.append(COUNTER) 
                        cv2.imwrite(ini_img_path + "/aligned_undetected_images/" + str(COUNTER) + '.jpg', faceAligned)
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

    #clustering the gathered data below
    ms = MeanShift(cluster_all=False)
    ms.fit(img_vector_data['vectorised_data'])
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    print(cluster_centers)
    uniq_labels = np.unique(labels)
    n_clusters_ = len(np.unique(labels))
    print("unique cluster labels:{}".format(labels))
    print("Number of labels calcualted:{}".format(len(labels)))
    #create each folder for each cluster
    create_child_dirs(uniq_labels, video_title_path + "/")
    #move images to respective dirs
    # for image_title in detected_counter:
    for i,label in enumerate(labels):
        move_file(ini_img_path+ "/detected_images/",video_title_path+"/"+str(label)+"/",str(detected_counter[i])+".jpg")

except Exception as e:
    pass
    raise e
else:
    pass
finally:
    pass
    # ms = MeanShift(cluster_all=False)
    # ms.fit(img_vector_data['vectorised_data'])
    # labels = ms.labels_
    # cluster_centers = ms.cluster_centers_
    # print(cluster_centers)
    # n_clusters_ = len(np.unique(labels))


    # csvData = []
    # csvData.append(["distance","angle"])
    # for d in img_vector_data['vectorised_data']:
    #     csvData.append(d)
    # file_name = video.split("/")[-1].split(".")[0] + ".csv"

    # with open(file_name, 'wb') as csvFile:
    #     writer = csv.writer(csvFile)
    #     writer.writerows(csvData) 
    # print(" labels:{}".format(labels))
    # print("Number of labels calcualted:{}".format(len(labels)))
    # print("unique labels generated:{}".format(np.unique(labels)))




