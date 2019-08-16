# usage: 
# python bulk_tester_snn.py --test-images kaggle_image_data/PrivateTest --model output/snn123.hdf5 --shape-predictor shape_predictor_68_face_landmarks.dat

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
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="path to input keras model file")
ap.add_argument("-i", "--test-images", required=True,
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

#initialize counters for Analysis
total_images = 0
total_correct = 0
total_wrong = 0

#csv headers
csvData = [['imageTitle', 'OriginalLabel','finalPrediction', 'full_predictions']]
csvData2 = ['Total_images_input','Total_images_calculated', 'total_correct', 'total_wrong', 'total_accuracy']
csvData3 = ['original_lable', 'no: of correct', 'no: of wrong', 'total_img_category', 'percentage_accuracy']
#
# Each array in below Dict has [total_correct, total_wrong, total_imgs_in_this_cat]
accuracy_dict = {"angry":[0,0,0], "disgust":[0,0,0], "fear":[0,0,0], "happy":[0,0,0], "sad":[0,0,0], "surprise":[0,0,0], "neutral":[0,0,0]}
# loading the trained NN model
print("[INFO] loading the pre-trained model for emotion prediction......")
emotion_classifier = load_model(args['model'], compile=False)

try:
    # loop over our testing images
    for imagePath in paths.list_images(args["test_images"]):
        # load the image, resize it to a fixed 32 x 32 pixels (ignoring
        # aspect ratio), and then extract features from it
        print("[INFO] classifying {}".format(
            imagePath[imagePath.rfind("/") + 1:]))
        url_split = imagePath.split('/')
        img_name, img_label, csvName = url_split[-1],emotions[int(url_split[-2])], url_split[-3]
        total_images += 1
        accuracy_dict[img_label][2] += 1
        image = cv2.imread(imagePath)
        frame = cv2.imread(imagePath)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = imutils.resize(gray, width=500)

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

            roi = image_to_feature_vector(gray)
            roi = roi.astype("float") / 255.0
            #roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label1 = emotions[preds.argmax()]

            label = "{}: {:.2f}%".format(emotions[preds.argmax()],
                emotion_probability * 100)
            if img_label == label1:
                total_correct += 1
                accuracy_dict[img_label][0] += 1
            else:
                total_wrong += 1  
                accuracy_dict[img_label][1] += 1  
            preds_label = {}
            for (i, emo) in enumerate(preds):
                preds_label[emotions[i]] = "{:.2f}%".format(preds[i] * 100)

            print("=========preds:{} and emotion_probability:{} and label:{}".format(preds,emotion_probability,label))
            # print("shape directly from faceutils nptoshape is:{}".format(type(shape)))
            # print("shape directly from faceutils nptoshape is length:{}".format(len(shape)))
            # Insert CSV data
            csvData.append([img_name,img_label,label,preds_label])

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

            cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 3)

            # show the output image with the face detections + facial landmarks
    #     cv2.imshow("Output", image)
    #     cv2.waitKey(0)
    #     if key == ord("q"):
    #         break
    # cv2.destroyAllWindows()
except Exception as e:
    raise e
else:
    pass
finally:
    
    csvData.append(csvData2)
    csvData.append([total_images, total_correct + total_wrong ,total_correct,total_wrong, 100 * float(total_correct)/float(total_correct + total_wrong)])
    csvData.append(csvData3)
    for key, value in accuracy_dict.items():
        csvData.append([key, value[0], value[1], value[2], 100 * float(value[0])/float(value[0] + value[1])])
    with open(csvName + '.csv', 'wb') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData) 

