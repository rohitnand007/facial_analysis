""" This Program is for taking the detected images and doing pca and then clustering them. 
# It does not process the video to and apply any kind of facial rec algo or take any screenshots of 
the image """

"""
*usage
python bucketwise_clustering.py --clustering kmeans --pca False --output_folder kmeans_mouth 
--data-folder data_folder/blink_test2 --bucket_path meanshift/0/
"""

# import the necessary packages
from helper_methods import *
from scipy.spatial import distance as dist
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imutils.face_utils import FACIAL_LANDMARKS_IDXS
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
ap.add_argument("-c", "--clustering", default=None, required=True,
    help="clustering algorithm to be specified")
ap.add_argument("-b", "--bucket_path", default=None, required=True,
    help="bucket path after data_folder for reclustering again")
ap.add_argument("-p", "--pca", default=True, 
    help="apply pca to the bucket again. Only True or False accepted")
ap.add_argument("-o", "--output_folder", default=None,
    help="give output folder name if you want something different")
ap.add_argument("-n","--cluster_count",default=15,
    help="number of clusters for kmeans algorithm")
args = vars(ap.parse_args())

#get mouth co-ordinates
mouth = FACIAL_LANDMARKS_IDXS['mouth']

#get mouth data_points from image_vector data
(mouth_start,mouth_end) = ((mouth[0]*2)-1, (mouth[1]*2)-1)

ini_data_path = args["data_folder"]
clustering_algo = args["clustering"]
bucket_path = args["bucket_path"]
pca_option = args["pca"]
print(pca_option)
output_folder = args["output_folder"] if args['output_folder'] != None else clustering_algo
print ini_data_path

# clustered_data_path = ini_data_path + "/"+ clustering_algo +"/" 

# if not os.path.exists(clustered_data_path): os.mkdir(clustered_data_path) 

video_file_name = ini_data_path.split('/')[-1]

detected_csv_file_path = ini_data_path + "/" + video_file_name + "_detected.csv"

undetected_csv_file_path = ini_data_path + "/" + video_file_name + "_mapping_and_undetected.csv"

detected_images_path = ini_data_path + "/" + bucket_path

clustered_data_path = detected_images_path +  output_folder +"/" 

if not os.path.exists(clustered_data_path): os.mkdir(clustered_data_path) 

sorted_images_array = sort_img_array(detected_images_path)

print(len(sorted_images_array))

print("Detected images are sorted successfully............................")

img_vector_data = {'vectorised_data':[]}
          
#create the parent directory for image storage:
# ini_img_path = os.getcwd() + "/data_folder/" + 
# print("===========================================.parent_dir created:{}".format(ini_img_path))

# loop over frames from the video stream
try:
    #import dict for the detected row and cooresponding image mapping
    mapping_data = {}
    with open(undetected_csv_file_path,'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if row[1] == "undetected_image_filename":
                break
            else:
                mapping_data[int(row[0].split(".")[0])] = int(row[1].split(".")[0])    
    print(mapping_data)
    #import the data from csv into an ord-dict
    current_img = iter(sorted_images_array)
    with open(detected_csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        img_name = next(current_img,-1)
        for row in csvreader:
            if img_name != -1:
                print("line_num{} : {} img_name getting printed here is".format((mapping_data[csvreader.line_num]),img_name))
                if (mapping_data[csvreader.line_num]) == int(img_name.split(".")[0]): 
                    img_vector_data['vectorised_data'].append([float(arr) for arr in row])
                    img_name = next(current_img,-1)
            else:
                break
                
        print("Data import from csv file finished")
    # converting array to np.float type
    converted_data = np.asarray(img_vector_data['vectorised_data']) #.astype(np.float64)
    converted_data = converted_data[:,mouth_start:mouth_end]
    converted_data_as_input = converted_data

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("number in folder: {} & number vectorised_: {}".format(len(sorted_images_array),len(img_vector_data['vectorised_data'])))

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    print("image data converted to numpy array..................................") 
    
    # Apply PCA transform to converted data
    # Scale using standard scalar on the datapoints
    if not pca_option=="False":
        sc = StandardScaler()
        sc.fit(converted_data)
        scaled_converted_data = sc.transform(converted_data)
        #initialize pca
        pca = PCA(.99)
        pca.fit(scaled_converted_data)
        new_coms = pca.n_components_
        pca_transformed_data = pca.transform(scaled_converted_data)
        converted_data_as_input = pca_transformed_data
        print("original shape:{}".format(scaled_converted_data.shape))
        print("pca transformed shape:{}".format(pca_transformed_data.shape))
        # print("output of PCA transform:{}".format(new_data))
        print("New number of components are:{} out of 136".format(new_coms))
        # print("feature contribution to pca:{}".format(pca.components_))
 

    #clustering the gathered data below
    print("clustering the data begins here.....................................")
    if clustering_algo == ("meanshift") or clustering_algo == "meanshift_mouth" :
        ms = MeanShift(cluster_all=False)
        ms.fit(converted_data_as_input)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        uniq_labels = np.unique(labels)
        n_clusters_ = len(np.unique(labels))
        print("unique cluster labels:{}".format(labels))
        print("Number of labels calcualted:{}".format(len(labels)))
        print("NUmber of unique labels calculated:{}".format(uniq_labels))

    elif clustering_algo == ("kmeans") or clustering_algo == "kmeans_mouth":
        cluster_count = args["cluster_count"]
        # ms = MeanShift(cluster_all=False)
        # ms.fit(pca_transformed_data)
        # labels = ms.labels_
        # n_clusters_ = len(np.unique(labels))
        km = KMeans(n_clusters=cluster_count).fit(converted_data_as_input)
        labels = km.labels_
        cluster_centers = km.cluster_centers_
        uniq_labels = np.unique(labels)
        n_clusters_ = len(np.unique(labels))
        print("unique cluster labels:{}".format(labels))
        print("Number of labels calcualted:{}".format(len(labels)))
        print("NUmber of unique labels calculated:{}".format(uniq_labels))
        print("clustering the data Ends here.....................................")
    elif clustering_algo == "something":
        pass
    else:
        pass        
        
    #create each folder for each cluster
    create_child_dirs(uniq_labels, clustered_data_path + "/")
    #move images to respective dirs
    # for image_title in sorted_images_array:
    for i,label in enumerate(labels):
       copy_file(detected_images_path,clustered_data_path +"/"+str(label)+"/",sorted_images_array[i])

# except Exception as e:
except ZeroDivisionError:
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





