"""This script used for testing out the clustering process with pca and mean shift in your local machine.
the better one is img_seg_v2_franklin.py .This is used as a kind of rough book for dev purposes
"""

# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
import pandas as pd
from scipy.spatial import distance as dist
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import math 
import numpy as np
import argparse
import time
import csv
import os

# jhgdyfgdgfudk
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--folder", 
    help="path to input folder containing euler files titles")
ap.add_argument("-u", '--ursi',
    help='ursi list of the files')
args = vars(ap.parse_args())
 
files_folder = args["folder"]

ursi_file = args['ursi']

files_list = []

ursi_list = []

with open(ursi_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        ursi_list.append(row[0])

uniq_list = set(ursi_list)
ursi_list = list(uniq_list)     

files_dict = {}

def get_files_list(files_folder):

    for root, dirs, files in os.walk(files_folder): 
        for f in files:
            if ((f.endswith('.csv')) and '_euler_angles' in f): 
                files_list.append(root+"/" + f)
    return files_list

all_files_list = get_files_list(files_folder)

def get_ursi_files_dict(files_list,ursi_list):
    for ursi in ursi_list:
        files_dict[ursi] = []
        for f in files_list:
            if str(ursi) in f:
                files_dict[ursi].append(f)
    return files_dict

def get_file_size(file_path):
    return os.path.getsize(file_path)   

def sort_files_by_size(ursi_dict):
    for key,value in ursi_dict.items():
        ursi_dict[key] = sorted(value,key=get_file_size,reverse=True)
    return ursi_dict

all_files_dict = get_ursi_files_dict(all_files_list,ursi_list)
sorted_files_dict = sort_files_by_size(all_files_dict)

print sorted_files_dict
print all_files_dict

for ursi,file_path_arr in sorted_files_dict.items():
    if len(file_path_arr) == 0: continue
    for csvfile in file_path_arr:
        ini_data_path = csvfile
        print ini_data_path

        csv_file_name = ini_data_path.split('/')[-1]

        csv_file_path = ini_data_path
        euler_vector_data = {'vectorised_data':[]}
                  
        # loop over frames from the video stream
        try:
            #import the data from csv into an ord-dict
            with open(csv_file_path, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                next(csvreader)
                for row in csvreader:
                    euler_vector_data['vectorised_data'].append([float(arr) for arr in row[1:]])
                print("Data import from csv file finished")

            # converting array to np.float type
            converted_data = np.asarray(euler_vector_data['vectorised_data'])

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
            pca_transformed_data = pca.transform(scaled_converted_data)
            print("original shape:{}".format(scaled_converted_data.shape))
            print("pca transformed shape:{}".format(pca_transformed_data.shape))
            # print("output of PCA transform:{}".format(new_data))
            print("New number of components are:{} out of 3".format(new_coms))
            print("feature contribution to pca:{}".format(pca.components_))

            tsne_transformed_data = TSNE().fit_transform(pca_transformed_data) #random_state=RS
            print("After PCA original shape:{}".format(pca_transformed_data.shape))
            print("tsne transformed shape:{}".format(tsne_transformed_data.shape))
            # print("output of PCA transform:{}".format(new_data))
            # print("New number of components are:{} out of 136".format(new_coms))
            converted_data_as_input = tsne_transformed_data
            print("Tsne completed")

            # clustering the gathered data below
            ms = MeanShift(cluster_all=False)
            ms.fit(tsne_transformed_data)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_
            print(cluster_centers)
            uniq_labels = np.unique(labels)
            n_clusters_ = len(np.unique(labels))
            print("unique cluster labels:{}".format(labels))
            print("Number of labels calcualted:{}".format(len(labels)))
            print("Number of unique labels calculated:{}".format(uniq_labels))

            #label counter:
            uniq_lable_counter = {}
            for num in uniq_labels:
                uniq_lable_counter[num] = 0
            for num in labels:
                uniq_lable_counter[num] += 1
            #end of label counter
            
            print("Uniq label counter dictionary:{} ".format(uniq_lable_counter))        
            
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

            # file1 =  open(csv_file_path.split(".")[0]+".txt", 'w+')
            file1 = open(os.getcwd()+"/cluster_txt/"+csv_file_path.split(".")[0].split("/")[-1]+".txt", 'w+')
            file1.write("Number of labels calcualted:{}".format(len(labels))+"\n") 
            file1.write("Number of unique labels calculated:{}".format(uniq_labels)+"\n")
            file1.write("Uniq label counter dictionary:{} ".format(uniq_lable_counter))

# hfhkdhfkdgd
 
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--csv", default=None, required=True,
#     help="path to csv file of euler angles")
# args = vars(ap.parse_args())

# ini_data_path = args["csv"]
# print ini_data_path

# csv_file_name = ini_data_path.split('/')[-1]

# csv_file_path = ini_data_path
# euler_vector_data = {'vectorised_data':[]}
          
# # loop over frames from the video stream
# try:
#     #import the data from csv into an ord-dict
#     with open(csv_file_path, 'r') as csvfile:
#         csvreader = csv.reader(csvfile)
#         next(csvreader)
#         for row in csvreader:
#             euler_vector_data['vectorised_data'].append([float(arr) for arr in row[1:]])
#         print("Data import from csv file finished")

#     # converting array to np.float type
#     converted_data = np.asarray(euler_vector_data['vectorised_data'])

#     print("converted_data is :{}".format(converted_data))

#     # PCA on this converted data:
#     # Scale using standard scalar on the datapoints
#     sc = StandardScaler()
#     sc.fit(converted_data)
#     scaled_converted_data = sc.transform(converted_data)
#     print(scaled_converted_data)
#     #initialize pca
#     pca = PCA(.99)
#     pca.fit(scaled_converted_data)
#     new_coms = pca.n_components_
#     pca_transformed_data = pca.transform(scaled_converted_data)
#     print("original shape:{}".format(scaled_converted_data.shape))
#     print("pca transformed shape:{}".format(pca_transformed_data.shape))
#     # print("output of PCA transform:{}".format(new_data))
#     print("New number of components are:{} out of 3".format(new_coms))
#     print("feature contribution to pca:{}".format(pca.components_))

#     tsne_transformed_data = TSNE().fit_transform(pca_transformed_data) #random_state=RS
#     print("After PCA original shape:{}".format(pca_transformed_data.shape))
#     print("tsne transformed shape:{}".format(tsne_transformed_data.shape))
#     # print("output of PCA transform:{}".format(new_data))
#     # print("New number of components are:{} out of 136".format(new_coms))
#     converted_data_as_input = tsne_transformed_data
#     print("Tsne completed")

#     # clustering the gathered data below
#     ms = MeanShift(cluster_all=False)
#     ms.fit(tsne_transformed_data)
#     labels = ms.labels_
#     cluster_centers = ms.cluster_centers_
#     print(cluster_centers)
#     uniq_labels = np.unique(labels)
#     n_clusters_ = len(np.unique(labels))
#     print("unique cluster labels:{}".format(labels))
#     print("Number of labels calcualted:{}".format(len(labels)))
#     print("Number of unique labels calculated:{}".format(uniq_labels))

#     #label counter:
#     uniq_lable_counter = {}
#     for num in uniq_labels:
#         uniq_lable_counter[num] = 0
#     for num in labels:
#         uniq_lable_counter[num] += 1
#     #end of label counter
    
#     print("Uniq label counter dictionary:{} ".format(uniq_lable_counter))        
    
# except Exception as e:
#     pass
#     raise e
# else:
#     pass
# finally:
#     pass
#     # ms = MeanShift(cluster_all=False)
#     # ms.fit(img_vector_data['vectorised_data'])
#     # labels = ms.labels_
#     # cluster_centers = ms.cluster_centers_
#     # print(cluster_centers)
#     # n_clusters_ = len(np.unique(labels))


#     # csvData = []
#     # csvData.append(["distance","angle"])
#     # for d in img_vector_data['vectorised_data']:
#     #     csvData.append(d)
#     # file_name = video.split("/")[-1].split(".")[0] + ".csv"

#     file1 =  open(csv_file_path.split(".")[0]+".txt", 'w+')
#     file1.write("Number of labels calcualted:{}".format(len(labels))+"\n") 
#     file1.write("Number of unique labels calculated:{}".format(uniq_labels)+"\n")
#     file1.write("Uniq label counter dictionary:{} ".format(uniq_lable_counter))




