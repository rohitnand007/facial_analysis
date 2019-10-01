# code snippet for searching for certain kind of videos
import os,glob
path = os.path.expanduser("~") + "/../../../export/research/analysis/human/kkiehl/media/"
addl_path = "BBP_20150/Assessment_Videos/NewMexico/Adult_Incarcerated/Male"

videos_array = glob.glob(path + addl_path +"/"+"**PCLR**.wmv")


#to make a python array to bash array

def bash_array(python_array):
	ini_string = ""
	for title in python_array:
		ini_string += '\"{}\" '.format(title)
	return ini_string.strip()

		