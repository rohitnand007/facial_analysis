# USAGE
# python snn1.py --dataset kaggle_image_data --model output/snn.hdf5

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os

def image_to_feature_vector(image, size=(48, 48)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model file")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

#initialize the emotions coded as numbers
emotions = {0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"sad", 5:"surprise", 6:"neutral"}

# initialize the data matrix and labels list
trainData, testData, trainLabels, testLabels = [], [], [], []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	path_data = imagePath.split(os.path.sep)
	dataset_type = path_data[-3].lower()
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	label = emotions[int(path_data[-2])]

	# construct a feature vector raw pixel intensities, then update
	# the data matrix and labels list
	features = image_to_feature_vector(gray)
	if dataset_type == 'training':
		trainData.append(features)
		trainLabels.append(label)
	elif dataset_type == 'privatetest':
		testData.append(features)
		testLabels.append(label)
	else:	
		trainData.append(features)
		trainLabels.append(label)
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# encode the labels, converting them from strings to integers
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.fit_transform(testLabels)

# scale the input image pixels to the range [0, 1], then transform
# the labels into vectors in the range [0, num_classes] -- this
# generates a vector for each label where the index of the label
# is set to `1` and all other entries to `0`
trainData = np.array(trainData) / 255.0
trainLabels = np_utils.to_categorical(trainLabels, 7)

# partition the data into training and testing plits, using 75%
# of the data for training and the remaining 25% for testing
testData = np.array(testData) / 255.0
testLabels = np_utils.to_categorical(testLabels, 7)

# print("[INFO] constructing training/testing split...")
# (trainData, testData, trainLabels, testLabels) = train_test_split(
# 	data, labels, test_size=0.25, random_state=42)

# define the architecture of the network
print("[INFO] constructing model with the kaggle dataset......")
model = Sequential()
model.add(Dense(1536, input_dim=2304, init="uniform",
	activation="relu"))
model.add(Dense(786, activation="relu", kernel_initializer="uniform"))
model.add(Dense(7))
model.add(Activation("softmax"))

# train the model using SGD
print("[INFO] compiling model...")
sgd = SGD(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
model.fit(trainData, trainLabels, epochs=100, batch_size=64,
	verbose=1)

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))

# dump the network architecture and weights to file
print("[INFO] dumping architecture and weights to file...")
model.save(args["model"])