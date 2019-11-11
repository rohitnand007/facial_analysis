# usage: 
# python video_tester_snn.py --video ./videos/test_video.mp4 --model snn123.hdf5 --shape-predictor ./shape_predictor_68_face_landmarks.dat

from imutils import face_utils
import dlib
import numpy as np
from keras.models import load_model
import time
import sys
import argparse
import cv2
import imutils
from keras.preprocessing.image import img_to_array
from imutils import paths
import os
import shutil
# from imutils.video import FileVideoStream
# from imutils.video import VideoStream
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="path to input keras model file")
ap.add_argument("-f", "--bucket_path", required=True,
    help="path to input number bucket ")
# ap.add_argument("-v", "--video", required=True,
#     help="path to input keras model file")
ap.add_argument("-s", "--shape-predictor", required=True,
    help="path to shape predictors")
args = vars(ap.parse_args())

def image_to_feature_vector(image, size=(48, 48)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size) #.flatten()

def create_child_dirs(dirs_array, parent_dir):
    if os.path.exists(parent_dir):
        for bucket in dirs_array:
            os.mkdir(parent_dir + str(bucket))
    else:
        print("No parent dir created......@@@") 

def get_all_images(imgs_path):
    tmp_array = []
    final_array = []
    for root,dirs,files in os.walk(imgs_path):
        for name in files:
            if name.endswith('.jpg'): final_array.append(imgs_path + name) 
        break    
    return final_array

def copy_file(image_path,label):
    path_split = image_path.rsplit("/",1)
    parent_dir, file_name = path_split[0], path_split[1]
    dest_dir = parent_dir + "/snn/" + str(label) + "/" 
    shutil.copy(image_path, dest_dir+file_name)       
 
# let's go code an faces detector(HOG) and after detect the 
# landmarks on this detected face

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

#Dictionary for emotion recognitiomodel output and emotions
emotions = {0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"sad", 5:"surprise", 6:"neutral"}

#create a parent directory for this Neural_net

new_parent_dir = args["bucket_path"] + "/" + "snn/"

if not os.path.exists(new_parent_dir): os.mkdir(new_parent_dir)

#create directories for each emotion in a different directory
create_child_dirs(emotions.keys(), new_parent_dir)


#initialize counters for Analysis
total_frames = 0
face_detected_frames = 0
predicted_frames = 0
emotions_counter = {"angry":0, "disgust":0, "fear":0, "happy":0, "sad":0, "surprise":0, "neutral":0}
angry = disgust = fear = happy = sad = surprise = neutral = True

# loading the trained NN model
print("[INFO] loading the pre-trained model for emotion prediction......")
emotion_classifier = load_model(args['model'], compile=False)

all_image_paths_array = get_all_images(args["bucket_path"])



try:
     # loop over our testing images
    for imagePath in all_image_paths_array:

        image = cv2.imread(imagePath)
        frame = cv2.imread(imagePath)
        print("image_path:{}".format(imagePath))

        total_frames +=1
        gray = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = imutils.rotate_bound(frame, 270)
        # gray = imutils.resize(gray, width=500)

        # detect faces in the grayscale image
        rects = detector(gray, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            # print("shape directly from predictor is:{}".format(shape))

            shape = face_utils.shape_to_np(shape)
            face_detected_frames += 1

            roi = image_to_feature_vector(gray)
            roi = roi.astype("float") / 255.0
            #roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_classifier.predict(roi)[0]
            print("preds:{}".format(preds))
            emotion_probability = np.max(preds)
            print("emotion_probability:{}".format(emotion_probability))
            label1 = preds.argmax()
            print("label1: {}".format(preds.argmax()))
            copy_file(imagePath,label1)

except Exception as e:
    raise e
else:
    pass
finally:
    file1 = open("results.txt","a")
    str1 = "total_frames:{} and emotions_counter:{} and face:{}..   /n".format(total_frames,emotions_counter,face_detected_frames)
    file1.write(str1)
    file1.close()
    print(str1) 
