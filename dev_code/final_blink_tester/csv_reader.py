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
with open('example.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV)
    for row in readCSV:
        full_array.append(row[1])
while i < len(full_array):
	sliced_array.append(full_array[i:i+2])
	i +=2

for f, b in zip(letters, sliced_array):
	new_array = ""
	for a in b:
		new_array += ' \"{}\"'.format(a)
		print(new_array)
	file1.write("{}=({})\n".format(f,new_array.strip()))        

file1.close()        