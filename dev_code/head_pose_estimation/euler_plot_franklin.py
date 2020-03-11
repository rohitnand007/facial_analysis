#importing essential modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.rotations import *
import argparse
import csv
import os

# construct the argument parse and parse the arguments
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


#we get sorted files according to ursi and file size until the abpve variable. we have to plot them in a new folder
ax = plot_basis(R=np.eye(3), ax_s=2)
axis = 0
angle = np.pi / 2
p = np.array([0.0, 0.0, 0.0])

def avg_rotation_mat(r):
	return [(r[0,0]+r[1,0]+r[2,0])/3.0,(r[0,1]+r[1,1]+r[2,1])/3.0,(r[0,2]+r[1,2]+r[2,2])/3.0]

def scatter_plot(data,img_name,path):
	intervals = len(data)/(15*300)
	data_start = 0
	if intervals >= 6:
		rangee = range(1,7)
	else:
		if int(intervals) == 0:
			rangee = [1]
		else:	
			rangee = range(1,int(intervals))

	for i in rangee:
		data_end = i*15*300
		if i==rangee[-1]:
			file_name = path + "full_" +img_name + '.png'
			data = data[data_start: data_end]
		else:	
			file_name = path + str(i*5)+"_min_" +img_name + '.png'
			data = data[data_start: data_end]
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		xs,ys,zs = [x[0] for x in data ],[x[1] for x in data ],[x[2] for x in data ]
		ax.scatter(xs, ys, zs, c='r', marker='.')
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')

		# plt.show()
		# plt.savefig("plots/" + "avg_rot_mat_scatter" + '.png')
		
		plt.savefig(file_name)
		print(file_name)


def parametric_curve(data,img_name,path):
	intervals = len(data)/(15*300)
	data_start = 0
	if intervals >= 6:
		rangee = range(1,7)
	else:
		if int(intervals) == 0:
			rangee = [1]
		else:	
			rangee = range(1,int(intervals))

	for i in rangee:
		data_end = i*15*300
		if i==rangee[-1]:
			file_name = path + "full_" +img_name + '.png'
			data = data[data_start:]
		else:	
			file_name = path + str(i*5)+"_min_" +img_name + '.png'
			data = data[data_start: data_end]
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		xs,ys,zs = [x[0] for x in data ],[x[1] for x in data ],[x[2] for x in data ]
		ax.plot(xs, ys, zs, label='parametric curve')
		ax.legend()

		# plt.show()
		# plt.savefig("plots/" + "avg_rot_mat_scatter" + '.png')
		
		plt.savefig(file_name)	
		print(file_name)
	

for ursi,file_path_arr in sorted_files_dict.items():
	if len(file_path_arr) == 0: continue
	path = os.getcwd() + "/plots/"
	csvdata = []
	plot_data = []
	euler_data = []
	j = 0
	print(file_path_arr[0])
	if not os.path.exists(path+ursi):
		os.mkdir(path + ursi)
		path += ursi + "/"
	with open(file_path_arr[0],'r') as csvfile:
		video_name = file_path_arr[0].split('/')[-1].split('.')[-2]
		csvreader = csv.reader(csvfile)
		next(csvreader)
		for row in csvreader:
			j +=1
			# ax = plot_basis(R=np.eye(3), ax_s=1)
			# ax = plot_basis()
			num_row = [float(i) for i in row]
			euler = [num_row[1],num_row[2],num_row[3]]
			euler_data.append(euler)
			# R = matrix_from_euler_xyz(euler)
			# K = avg_rotation_mat(R)
			# plot_data.append(K)
	scatter_plot(euler_data,"euler_scatter_"+video_name,path)
	parametric_curve(euler_data,"euler_parametric_"+video_name,path)		



