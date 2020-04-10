#inclues helper methods for image_segregation.py
import os, shutil
import math
from imutils.face_utils import FACIAL_LANDMARKS_IDXS
import pandas as pd
from scipy.spatial import distance as dist
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np
import cv2


def create_parent_dir(video_title_path):
    video_title_path = video_title_path
    if not os.path.exists(video_title_path):
        os.makedirs(video_title_path + '/img_temp' + '/detected_images') 
        os.makedirs(video_title_path + '/img_temp' + '/undetected_images') 
        # os.makedirs(video_title_path + '/img_temp' + '/aligned_undetected_images')
    return video_title_path + '/img_temp/'   

def create_child_dirs(dirs_array, parent_dir):
    if os.path.exists(parent_dir):
        for bucket in dirs_array:
            os.mkdir(parent_dir + str(bucket))
    else:
        print("No parent dir created......@@@") 

def move_file(parent_dir,dest_dir,file_name):
    os.rename(parent_dir+file_name, dest_dir+file_name) 

def copy_file(parent_dir,dest_dir,file_name):
    shutil.copy(parent_dir+file_name, dest_dir+file_name)

def sort_img_array(imgs_path):
    tmp_array = []
    final_array = []
    for root,dirs,files in os.walk(imgs_path):
        for name in files:
            if name.endswith('.jpg'): tmp_array.append(int(name.split(".")[0])) 
        break    
    tmp_array.sort()
    for num in tmp_array:        
        final_array.append(str(num) + ".jpg")
    return final_array      

#method for getting head pose
def get_head_pose(shape):
    # initiators for assigning default values for this program
    K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
         0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
         0.0, 0.0, 1.0]
    D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

    cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
    dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

    object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                             [1.330353, 7.122144, 6.903745],
                             [-1.330353, 7.122144, 6.903745],
                             [-6.825897, 6.760612, 4.402142],
                             [5.311432, 5.485328, 3.987654],
                             [1.789930, 5.393625, 4.413414],
                             [-1.789930, 5.393625, 4.413414],
                             [-5.311432, 5.485328, 3.987654],
                             [2.005628, 1.409845, 6.165652],
                             [-2.005628, 1.409845, 6.165652],
                             [2.774015, -2.080775, 5.048531],
                             [-2.774015, -2.080775, 5.048531],
                             [0.000000, -3.116408, 6.097667],
                             [0.000000, -7.415691, 4.070434]])

    reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                               [10.0, 10.0, -10.0],
                               [10.0, -10.0, -10.0],
                               [10.0, -10.0, 10.0],
                               [-10.0, 10.0, 10.0],
                               [-10.0, 10.0, -10.0],
                               [-10.0, -10.0, -10.0],
                               [-10.0, -10.0, 10.0]])

    line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                  [4, 5], [5, 6], [6, 7], [7, 4],
                  [0, 4], [1, 5], [2, 6], [3, 7]] 


    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle

def compare_euler_angles(interval,ref_frame,actual_frame):
    diff_x,diff_y,diff_z = abs(ref_frame[0]-actual_frame[0]),abs(ref_frame[1]-actual_frame[1]),abs(ref_frame[2]-actual_frame[2])    
    return 1 if (diff_x > interval or diff_y > interval or diff_z > interval) else 0

def check_for_head_movement(dict_of_values):
    print(dict_of_values)
    det_frames, moved_frames = dict_of_values.keys()[-1], sum(dict_of_values.values())
    return 1 if moved_frames >= int(det_frames/2) else 0 

def meanshift_clustering_after_pca_tsne(data):
    converted_data = np.asarray(data)
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

    uniq_lable_counter = {}
    for num in uniq_labels:
        uniq_lable_counter[num] = 0
    for num in labels:
        uniq_lable_counter[num] += 1

    return uniq_labels,labels, uniq_lable_counter   
