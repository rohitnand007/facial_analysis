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
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="path to input keras model file")
ap.add_argument("-v", "--video", required=True,
    help="path to input video ")
ap.add_argument("-s", "--shape-predictor", required=True,
    help="path to shape predictors")
args = vars(ap.parse_args())

def image_to_feature_vector(image, size=(48, 48)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size) #.flatten()
 
# let's go code an faces detector(HOG) and after detect the 
# landmarks on this detected face

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

#Dictionary for emotion recognitiomodel output and emotions
emotions = {0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"sad", 5:"surprise", 6:"neutral"}

#initialize counters for Analysis
total_frames = 0
face_detected_frames = 0
predicted_frames = 0
emotions_counter = {"angry":0, "disgust":0, "fear":0, "happy":0, "sad":0, "surprise":0, "neutral":0}
angry = disgust = fear = happy = sad = surprise = neutral = True

# loading the trained NN model
print("[INFO] loading the pre-trained model for emotion prediction......")
emotion_classifier = load_model(args['model'], compile=False)

# start the video stream thread
print("[INFO] starting video stream thread...")
#vs = FileVideoStream(args["video"]).start()
vs = cv2.VideoCapture(args["video"])
fileStream = True
fps = vs.get(cv2.CAP_PROP_FPS)


try:
    while True:
        # if fileStream and not vs.more():
        #     break

        _,frame = vs.read()
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
            emotion_probability = np.max(preds)
            label1 = emotions[preds.argmax()]

            label = "{}: {:.2f}%".format(emotions[preds.argmax()],
                emotion_probability * 100)
            emotions_counter[label1] += 1
            print("=========preds:{} and emotion_probability:{} and label:{}".format(preds,emotion_probability,label))

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(gray, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(gray, (x, y), 1, (0, 0, 255), -1)

            cv2.putText(gray, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 3)    
        #print out the first emotion detected with probablility greater than 85%
        if emotions_counter[label1] < 5 and vars()[label1] and emotion_probability > 0.80:
            cv2.imwrite("output_pictures/"+ label1 +"time.strftime("%Y%m%d-%H%M%S").jpg", gray)
            vars()[label1] = False

except Exception as e:
    raise e 
else:
    pass
finally:
    file1 = open("results.txt","a")
    str1 = "total_frames:{} and emotions_counter:{} and face:{}..   /n".format(total_frames,emotions_counter,face_detected_frames)
    str2 = "Frame rate for this camera is :{}".format(fps)
    file1.write(str1)
    file1.write(str2)
    file1.close()
    print(str1) 
