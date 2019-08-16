# usage: 
# python demo_snn.py --image test1.jpg --model output/snn1.hdf5 --shape-predictor shape_predictor_68_face_landmarks.dat

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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="path to input keras model file")
ap.add_argument("-i", "--image", required=True,
    help="path to input image ")
ap.add_argument("-s", "--shape-predictor", required=True,
    help="path to shape predictors")
args = vars(ap.parse_args())

def image_to_feature_vector(image, size=(48, 48)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()
 
# let's go code an faces detector(HOG) and after detect the 
# landmarks on this detected face

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
# p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

#Dictionary for emotion recognitiomodel output and emotions
emotions = {0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"sad", 5:"surprise", 6:"neutral"}

# loading the trained NN model
print("[INFO] loading the pre-trained model for emotion prediction......")
emotion_classifier = load_model(args['model'], compile=False)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
frame = cv2.imread(args["image"])
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = imutils.resize(gray, width=500)

# detect faces in the grayscale image
rects = detector(gray, 1)

# print("######################rects:{}".format(rects))
# print("dtype is {}".format(type(rects)))
# print("len is {}".format(len(rects)))

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    # print("shape directly from predictor is:{}".format(shape))

    shape = face_utils.shape_to_np(shape)

    roi = image_to_feature_vector(gray)
    roi = roi.astype("float") / 255.0
    # roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = emotions[preds.argmax()]
    label = "{}: {:.2f}%".format(emotions[preds.argmax()],
        emotion_probability * 100)

    print("=========preds:{} and emotion_probability:{} and label:{}".format(preds,emotion_probability,label))
    # print("shape directly from faceutils nptoshape is:{}".format(type(shape)))
    # print("shape directly from faceutils nptoshape is length:{}".format(len(shape)))


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

# show the output image with the face detections + facial landmarks
cv2.imshow("Output", gray)
cv2.waitKey(0)