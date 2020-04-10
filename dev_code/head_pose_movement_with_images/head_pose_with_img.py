# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from helper_methods import *
import pandas as pd
from scipy.spatial import distance as dist
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import math 
from imutils import face_utils
# from imutils.face_utils import FaceAligner
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

euler_vector_data = {'vectorised_data':[]}
csvData = []

COUNTER = 0  
# measure_1 is for calulating scaling values of frame 1 and storing it
measure_x = -1
measure_y = -1
detected_counter = []
undetected_counter = []          
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])
# fa = FaceAligner(predictor, desiredFaceWidth=256)

print(detector)

# start the video stream thread
print("[INFO] starting video stream thread...")
video = args["video"]
video_title_path = os.path.expanduser("~") + "/../../../export/research/analysis/human/kkiehl/shared/Projects/VideoAnalysis/head_pose_clustered_images_latest" + "/" +  video.split("/")[-1].split(".")[0]
# video_title_path = os.getcwd() + "/data_folder" + "/" +  video.split("/")[-1].split(".")[0]
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
            frame = imutils.rotate_bound(frame, 0)
            #increase the frame counter by 1
            COUNTER += 1

            # frame = imutils.rotate_bound(frame, 270)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale frame
            rects = detector(gray, 0)  
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
            print("+++++++++++++++++++++++++++>{}".format(COUNTER))
            # shape = predictor(gray, rect)
            # faceAligned = fa.align(frame, gray, rect)
            # grayAlined = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
            if len(rects) > 0:
                for rect in rects:
                    cv2.imwrite(ini_img_path + "/detected_images/" + str(COUNTER) + '.jpg', frame)
                    shape = predictor(gray,rect)
                    shape = face_utils.shape_to_np(shape)
                    reprojectdst, euler_angle = get_head_pose(shape)
                    x_dist, y_dist, z_dist = euler_angle[0,0], euler_angle[1,0], euler_angle[2,0]
                    euler_vector_data['vectorised_data'].append([x_dist,y_dist,z_dist]) 
                    csvData.append([COUNTER,x_dist,y_dist,z_dist]) 
                    # (x, y, w, h) = face_utils.rect_to_bb(rectAlined)
                    # cv2.rectangle(faceAligned, (x, y), (x + w, y + h), (0, 255, 0), 2)
                          
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
    # if not os.path.exists(out_path +"/"+just_video_name):
    #         os.makedirs(out_path + "/"+just_video_name)

    csvData.insert(0,["frameNumber","x_angle", "y_angle", "z_angle"])
    file_name = video_title_path+"/"+ video.split("/")[-1].split(".")[0]+ "_euler_angles.csv"

    with open(file_name, 'wb') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData) 

    #meanshift clustering after pca+tsne
    uniq_labels,labels,uniq_lable_counter = meanshift_clustering_after_pca_tsne(euler_vector_data['vectorised_data'])

    #rearranging the deetcted images
    detected_images_path = ini_img_path + "/detected_images/"
    clustered_data_path = detected_images_path +  "meanshift/" 
    if not os.path.exists(clustered_data_path): os.mkdir(clustered_data_path)

    sorted_images_array = sort_img_array(detected_images_path)
    create_child_dirs(uniq_labels, clustered_data_path + "/")
    #move images to respective dirs
    # for image_title in sorted_images_array:
    for i,label in enumerate(labels):
       move_file(detected_images_path,clustered_data_path +"/"+str(label)+"/",sorted_images_array[i])

    #writing labels counter to a file    

    file1 = open(video_title_path+"/"+ video.split("/")[-1].split(".")[0]+ "_euler_angles+.txt", 'w+')
    file1.write("Number of labels calcualted:{}".format(len(labels))+"\n") 
    file1.write("Number of unique labels calculated:{}".format(uniq_labels)+"\n")
    file1.write("Uniq label counter dictionary:{} ".format(uniq_lable_counter))
    file1.close() 
  





