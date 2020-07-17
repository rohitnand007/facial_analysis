import os
import argparse
import csv
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--folder", required=True,
	help="path to input folder of videos")
args = vars(ap.parse_args())

files_folder = args["folder"]

files_list = []
blah = []

def get_files_list(files_folder):

	for root, dirs, files in os.walk(files_folder):
			for f in files:
				if (f.endswith('.wmv') or f.endswith('.mp4')): 
					files_list.append(root + f)
	return files_list

def get_files_list_only_pclr(files_folder):

	for root, dirs, files in os.walk(files_folder):
			for f in files:
				if ((f.endswith('.wmv') or f.endswith('.mp4')) and 'PCLR' in f): 
					files_list.append(root + f)
	return files_list

def get_files_list_only_skid(files_folder):
	for root, dirs, files in os.walk(files_folder):
			for f in files:
				if ((f.endswith('.wmv') or f.endswith('.mp4')) and 'SCID' in f): 
					files_list.append(root + f)
	return files_list



a = get_files_list(files_folder)

file_chunks = [a[x:x+50] for x in xrange(0, len(a), 50)]

for i,chunk in enumerate(file_chunks):
	output_filename = 'output_csvs/juve_pose_list_' + str(i) + ".csv" 
	with open(output_filename, 'w') as file:
		writer = csv.writer(file)
		for name in chunk:	
			writer.writerow([name])

# with open('pose_list_0.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#     	blah.append(row[0])

# print blah    	


						