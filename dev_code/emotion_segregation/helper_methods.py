#inclues helper methods for image_segregation.py
import os
import math
from imutils.face_utils import FACIAL_LANDMARKS_IDXS
from scipy.spatial import distance as dist
import numpy as np

def get_measure_y(shape):
    mx = dist.euclidean(shape[27],shape[28]) + dist.euclidean(shape[28],shape[29])
    return mx

def get_measure_x(shape):
    # extract the left and right eye (x, y)-coordinates
    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]
    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
    eyes_distance = dist.euclidean(leftEyeCenter,rightEyeCenter)
    return eyes_distance


def get_dist_angle(shape, measure_x, measure_y):
    distance = 0
    angle = 0
    data_array = []
    measure_nx = get_measure_x(shape)
    measure_ny = get_measure_y(shape)

    scale_x = measure_x / measure_nx
    scale_y = measure_y / measure_ny
    print("scaler value............................{},{}".format(scale_x,scale_y))
    (xmean, ymean) = np.mean(shape[:,0]), np.mean(shape[:,1])
    for point in shape:
        (x,y) = point[0],point[1]
        # distance = dist.euclidean((xmean,ymean), (x,y)) * scale
        #compute distance with scaling included
        dx, dy = (xmean -x)/scale_x , (ymean - y)/scale_y
        distance = np.sqrt((dx ** 2) + (dy ** 2))
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