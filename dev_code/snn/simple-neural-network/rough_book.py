# from imutils import paths
# from sklearn.preprocessing import LabelEncoder
# from keras.utils import np_utils
import numpy as np
import argparse
#from imutils import paths
# import cv2
import os

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())

a = list_images(args['dataset'])

# a is a generator expression(lazy list comprension
# one value of a is as: "dos_cats/kaggle_dogs_vs_cats/train/cat.4989.jpg" 

imagePaths = list(list_images(args["dataset"]))

#imagePaths gives a list of all image paths

# initialize the data matrix and labels list
data = []
labels = []

a = imagePaths[100]
print a

# image = cv2.imread(a)

emotions = {0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"sad", 5:"surprise", 6:"neutral"}

label = a.split(os.path.sep)[-3].lower()
# features = image_to_feature_vector(image)


print("==============================={}".format(label))
#===============================cat
print("==============================={}".format(features))
#===============================[ 85  99 110 ... 124 135 143]

# data.append(features)
# labels.append(label)
# labels.append("dog")
# labels.append("cat")

# le = LabelEncoder()
# labels = le.fit_transform(labels)

# print("+++++++++++++++++++++++++++++++++++++{},{},{},{}".format(labels,type(labels),len(labels),labels[-1]))

# #+++++++++++++++++++++++++++++++++++++[0 1 2],<type 'numpy.ndarray'>,3,2

# data = np.array(data) / 255.0

# print("============================{}".format(data))

# labels = np_utils.to_categorical(labels, 2)

# print("============================{}".format(labels))
# [[1. 0.]
#  [0. 1.]
#  [1. 0.]]







