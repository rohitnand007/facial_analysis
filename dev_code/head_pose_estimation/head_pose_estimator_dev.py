# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

#Note: angular sideways is x_dist, straight up and down is y_dist, z_dist is lateral side movement.

# import the necessary packages
from fps_calculator import fps_calculator
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import FileVideoStream
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import csv
import os

def collect_output_dir(path):
	# path = "/na/homes/ryerramsetty/../../../export/research/analysis/human/kkiehl/media/BBP_20150/assessment_videos/Wisconsin/Incarcerated_juvenile/video.wmv"
	a = path.split("/")
	# a = a[13:]
	# a = a[2:]
	del a[-1]
	# a.append(just_video_name)
	return a

def create_child_dirs(dirs_array, parent_dir):
    if os.path.exists(parent_dir):
        parent_directory = parent_dir
        for bucket in dirs_array:
        	parent_directory += str(bucket)
        	print(parent_directory)
        	if not os.path.exists(parent_directory):
	            os.mkdir(parent_directory)  
    		parent_directory += "/"
        return parent_directory    
    else:
        print("No parent dir created......@@@")	

def is_valid_sec(fraction,det_frame_count,fps):
	return True if det_frame_count >= int(fraction * fps) else False
	
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", default=None,
    help="path to input video ")
ap.add_argument("-d", "--csv", 
	help="path to input csv file containing video titles")
args = vars(ap.parse_args())

# Directory walk to grab all files.
videos = []
if args["video"] is None:
	with open(args["csv"]) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			videos.append(row[0])			
else:
	videos = [args["video"]]

print(videos)	
print(len(videos))

video_count = 0	

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


#method for getting head pose
def get_head_pose(shape):
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
	diff_x,diff_y = abs(ref_frame[1]-actual_frame[1]),abs(ref_frame[2]-actual_frame[2])	
	return 1 if (diff_x < interval or diff_y < interval) else 0

def check_for_head_movement(dict_of_values):
	print(dict_of_values)
	det_frames, moved_frames = dict_of_values.keys()[-1], sum(dict_of_values.values())
	return 1 if moved_frames >= int(det_frames/2) else 0 
	
for video in videos:
	video_count += 1
	# create the output directory with same tree structure as input video path
	just_video_name = video.split("/")[-1].split(".")[0]
	output_result_path = os.path.expanduser("~") + "/../../../export/research/analysis/human/kkiehl/shared/Projects/VideoAnalysis/BlinkAnalysis/juve_blinks_data/"
	#output_result_path = os.path.expanduser("~") + "/mrn_dev/facial_analysis/dev_code/head_pose_estimation/" + "/test_dir/"
	dirs_array = collect_output_dir(video) 
	out_path =  create_child_dirs(dirs_array,output_result_path)

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])

	# start the video stream thread
	# video = args["video"]
	fps = fps_calculator(video)
	print("=======================================================:{}".format(fps))
	print("[INFO] starting video stream thread...")
	vs = cv2.VideoCapture(video) 
	#vs = FileVideoStream(args["video"]).start()
	#time.sleep(1.0)
	#initializing counters and vars for calculation here
	frame_counter, frames_in_sec, current_sec, total_sec = 0,0,0,0
	detected_frames,total_detected_frames,total_frame_counter = 0,0,0
	ref_frame = False
	euler_angles_in_current_sec = {}
	csvData = []
	csvData1 = []

	# loop over frames from the video stream
	try:
		while True:
		# if this is a file video stream, then we need to check if
		# there any more frames left in the buffer to process
			(grabbed,frame) = vs.read()
			# print(grabbed)
			if grabbed:
				# grab the frame from the threaded video file stream, resize
				# it, and convert it to grayscale channels)

				frame = imutils.resize(frame, width=450)
				#frame = imutils.rotate_bound(frame, 270)
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				frame_counter += 1
				frames_in_sec += 1
				if (frames_in_sec - 1) == int(fps):
					total_sec += 1
					current_sec += 1
					# frames_in_sec = 1
				#frame = imutils.rotate_bound(frame, 90)
				# img_name = "junk_images/opencv_frame_{}.png".format(total_frame_counter)
	  			#cv2.imwrite(img_name, frame)
				# detect faces in the grayscale frame
				rects = detector(gray, 0)  	
				if len(rects) >= 1:
					detected_frames += 1
					total_detected_frames += 1
					print("incremented.........................................")
					print(detected_frames)
					# loop over the face detections
					for rect in rects:
						# determine the facial landmarks for the face region, then
						# convert the facial landmark (x, y)-coordinates to a NumPy
						# array
						shape = predictor(gray, rect)
						shape = face_utils.shape_to_np(shape)
						reprojectdst, euler_angle = get_head_pose(shape)
						x_dist, y_dist, z_dist = euler_angle[0,0], euler_angle[1,0], euler_angle[2,0]

						#setting the reference frame for the second
						if not ref_frame:
							ref_frame = (x_dist,y_dist,z_dist)

						euler_angles_in_current_sec[detected_frames] = compare_euler_angles(10,ref_frame,(x_dist,y_dist,z_dist))	

						csvData1.append([frame_counter,x_dist,y_dist,z_dist])

				if (current_sec == 1 and detected_frames >= int(0.6 * frames_in_sec)):
					cal_actual_sec = int(frame_counter/fps)
					csvData.append([cal_actual_sec,check_for_head_movement(euler_angles_in_current_sec)])
					current_sec = 0	
					ref_frame = False
					euler_angles_in_current_sec = {}
					detected_frames = 0
					frames_in_sec = 1
				elif (current_sec == 1 and detected_frames < int(0.6 * frames_in_sec)):
					cal_actual_sec = int(frame_counter/fps)
					csvData.append([cal_actual_sec, -1])
					current_sec = 0	
					ref_frame = False
					euler_angles_in_current_sec = {}
					detected_frames = 0	
					frames_in_sec = 1	


				# 		for (x, y) in shape:
				# 			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

				# 		for start, end in line_pairs:
				# 			cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

				# 		cv2.putText(frame, "X: " + "{:7.2f}".format(x_dist), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
				# 		            0.75, (0, 0, 0), thickness=2)
				# 		cv2.putText(frame, "Y: " + "{:7.2f}".format(y_dist), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
				# 		            0.75, (0, 0, 0), thickness=2)
				# 		cv2.putText(frame, "Z: " + "{:7.2f}".format(z_dist), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
				# 		            0.75, (0, 0, 0), thickness=2)
				# cv2.imshow("demo", frame)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

								

			else:
				break
		# do a bit of cleanup
		cv2.destroyAllWindows()
		# vs.stop()		

	except Exception as e:
		pass
		raise e
	else:
		pass
	finally:
		pass
		# csvData.insert(0,["frame_num","x", "y", "z"])
		if not os.path.exists(out_path +"/"+just_video_name):
			os.makedirs(out_path + "/"+just_video_name)

		csvData.insert(0,["time_in_sec", "head_movement"])
		file_name = out_path + "/" + just_video_name+"/"+ just_video_name + "_head_movement" + ".csv"

		csvData1.insert(0,["frameNumber","x_angle", "y_angle", "z_angle"])
		file_name1 = out_path + "/" + just_video_name+"/"+ just_video_name + "_euler_angles.csv"

		with open(file_name, 'wb') as csvFile:
			writer = csv.writer(csvFile)
			writer.writerows(csvData) 

		with open(file_name1, 'wb') as csvFile:
			writer = csv.writer(csvFile)
			writer.writerows(csvData1)	

		file1 = open("processed_videos_list.txt","a")
		file1.write(str(video_count) + ":" +video)
		file1.write("\n")
		file1.close()	



	