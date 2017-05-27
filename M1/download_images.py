from google.cloud import storage as gcs
import csv
from numpy import genfromtxt
import requests
import tensorflow as tf


#get the bucket and blob containing the .csv data file
client=gcs.Client()
try:
	bucket=client.get_bucket('parquery-sandbox')
except google.cloud.exceptions.NotFound:
	print('Sorry, that bucket does not exist!')

#uncomment to display bucket properties
#print(bucket.client)
#print(bucket)

#uncomment to debug blobs
#iterator=bucket.list_blobs()
#for i in iterator:
#	print(i)
	
try:
	blob=bucket.get_blob('treedom/query_result.csv')
except google.cloud.exceptions.NotFound:
	print('Sorry, that blob does not exist!')

#uncomment to display blob properties
#print(blob)	
#print(blob.client)

blob.download_to_filename('data.csv')

with open('data.csv', 'rb') as csvfile:
	#mydata=genfromtxt(csvfile, delimiter=',')
	readCSV=csv.reader(csvfile, delimiter=',')
	labels=[]
	links=[]
	for row in readCSV:
		label=row[5]
		link=row[7]
		
		if link != "NULL":
			labels.append(label)
			links.append(link)

#remove first element from array - the header data
labels.pop(0)
links.pop(0)

labels=map(int, labels)

print(labels[0])
print(links[0].split('/'))
print(links[0].split('/')[5])

correct_folder='correct/'
incorrect_folder='incorrect/'

#downloading images for training
for i in range(10000):
	if(labels[i] == 1):
		folder=correct_folder
	else:
		folder=incorrect_folder
		
	image=requests.get(links[i]).content
	
	with open('images/training/' + folder + links[i].split('/')[5] + '.jpg', 'wb') as handler:
		handler.write(image)
		
#downloading images for testing
for i in range(10000, 12000):
	if(labels[i] == 1):
		folder=correct_folder
	else:
		folder=incorrect_folder
		
	image=requests.get(links[i]).content
	with open('images/testing/' + folder + links[i].split('/')[5] + '.jpg', 'wb') as handler:
		handler.write(image)