#inclues helper methods for image_segregation.py
import os
import math
from scipy.spatial import distance as dist
import numpy as np

def get_measure(shape):
    m = dist.euclidean(shape[27],shape[28]) + dist.euclidean(shape[28],shape[29])
    return m

def get_dist_angle(shape, measure_1):
    distance = 0
    angle = 0
    data_array = []
    measure_n = get_measure(shape)
    scale = measure_1 / measure_n
    print("scaler value............................{}".format(scale))
    (xmean, ymean) = np.mean(shape[:,0]), np.mean(shape[:,1])
    for point in shape:
        (x,y) = point[0],point[1]
        distance = dist.euclidean((xmean,ymean), (x,y)) * scale
        xcentral, ycentral = (x-xmean),(y-ymean) 
        angle = (math.atan2(ycentral, xcentral)*360)/(2*math.pi)
        data_array.append([distance,angle])
        # data_array.append([distance])
        flat_array = [item for sublist in data_array for item in sublist]
    return flat_array

def create_parent_dir(video_title_path):
    video_title_path = video_title_path
    if not os.path.exists(video_title_path):
        os.makedirs(video_title_path + '/img_temp' + '/detected_images') 
        os.makedirs(video_title_path + '/img_temp' + '/undetected_images') 
        os.makedirs(video_title_path + '/img_temp' + '/alined_undetected_images')
    return video_title_path + '/img_temp/'   

def create_child_dirs(dirs_array, parent_dir):
    if os.path.exists(parent_dir):
        for bucket in dirs_array:
            os.mkdir(parent_dir + str(bucket))
    else:
        print("No parent dir created......@@@") 

def move_file(parent_dir,dest_dir,file_name):
    os.rename(parent_dir+file_name, dest_dir+file_name)   