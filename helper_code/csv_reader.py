import csv
import string
i=0
full_array = []
letters = []
sliced_array = []
file1 = open("video_files_list.txt","w")
for a in list(string.letters[26:]):
	b = a + "RR"
	letters.append(b)
print(letters)	
with open('FileListForBlinkAnalysis.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV)
    for row in readCSV:
        full_array.append(row[3])
while i < len(full_array):
	sliced_array.append(full_array[i:i+10])
	i +=10

for f, b in zip(letters, sliced_array):
	new_array = ""
	for a in b:
		new_array += ' \"{}\"'.format(a)
		print(new_array)
	file1.write("{}=({})\n".format(f,new_array.strip()))        

file1.close()        