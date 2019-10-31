"""This script used for testing out the clustering process with pca and mean shift in your local machine.
the better one is img_seg_v2_franklin.py .This is used as a kind of rough book for dev purposes
"""

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
import math 
import numpy as np
import argparse
import time
import csv
import os
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--data-folder", default=None, required=True,
    help="path to csv data and images folder")
args = vars(ap.parse_args())

ini_data_path = args["data_folder"]
print ini_data_path

video_file_name = ini_data_path.split('/')[-2]

csv_file_path = ini_data_path + "/" + video_file_name + "_detected.csv"

detected_images_path = ini_data_path + "/img_temp/detected_images/"

sorted_images_array = sort_img_array(detected_images_path)

print(sorted_images_array)

img_vector_data = {'vectorised_data':[]}
          
#create the parent directory for image storage:
# ini_img_path = os.getcwd() + "/data_folder/" + 
# print("===========================================.parent_dir created:{}".format(ini_img_path))

# loop over frames from the video stream
try:
    #import the data from csv into an ord-dict
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            img_vector_data['vectorised_data'].append([float(arr) for arr in row])
        print("Data import from csv file finished")

    # converting array to np.float type
    converted_data = np.asarray(img_vector_data['vectorised_data'])

    print("converted_data is :{}".format(converted_data))

    # PCA on this converted data:
    # Scale using standard scalar on the datapoints
    sc = StandardScaler()
    sc.fit(converted_data)
    scaled_converted_data = sc.transform(converted_data)
    print(scaled_converted_data)
    #initialize pca
    pca = PCA(.99)
    pca.fit(scaled_converted_data)
    new_coms = pca.n_components_
    new_data = pca.transform(scaled_converted_data)
    print("original shape:{}".format(scaled_converted_data.shape))
    print("pca transformed shape:{}".format(new_data.shape))
    # print("output of PCA transform:{}".format(new_data))
    print("New number of components are:{} out of 136".format(new_coms))
    print("feature contribution to pca:{}".format(pca.components_))

    # clustering the gathered data below
    ms = MeanShift(cluster_all=False)
    ms.fit(converted_data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    print(cluster_centers)
    uniq_labels = np.unique(labels)
    n_clusters_ = len(np.unique(labels))
    print("unique cluster labels:{}".format(labels))
    print("Number of labels calcualted:{}".format(len(labels)))
    print("NUmber of unique labels calculated:{}".format(uniq_labels))
    create each folder for each cluster
    create_child_dirs(uniq_labels, ini_data_path + "/")
    #move images to respective dirs
    # for image_title in sorted_images_array:
    for i,label in enumerate(labels):
       copy_file(detected_images_path,ini_data_path+"/"+str(label)+"/",sorted_images_array[i])

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




