import google.cloud.storage as gcs
import csv
import requests

import config as cfg
import tensorflow as tf
import os

#get the bucket and blob containing the .csv data file
client=gcs.Client()
try:
	bucket=client.get_bucket('parquery-sandbox')
except google.cloud.exceptions.NotFound:
	print('Sorry, that bucket does not exist!')
	
try:
	blob=bucket.get_blob('treedom/query_result.csv')
except google.cloud.exceptions.NotFound:
	print('Sorry, that blob does not exist!')

blob.download_to_filename('data.csv')

with open('data.csv', 'rb') as csvfile:

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

#downloading images for training
for i in range(cfg.dataset['training_set_size']):
	
	folder = cfg.dataset['label' + str(labels[i]) + '_folder']
		
	image=requests.get(links[i]).content
	
	directory = 'train-' + str(cfg.dataset['training_set_size']) + '-test-' + str(cfg.dataset['testing_set_size']) + '/' + cfg.directory['training'] + '/' + folder
	if not os.path.exists(directory):
		os.makedirs(directory)	
	
	with open( directory +  '/' + links[i].split('/')[5] + '.jpg', 'wb') as handler:
		handler.write(image)
		
#downloading images for testing
for i in range(10000, 10000 + cfg.dataset['testing_set_size']):
	
	folder = cfg.dataset['label' + str(labels[i]) + '_folder']
		
	image=requests.get(links[i]).content
	
	directory = 'train-' + str(cfg.dataset['training_set_size']) + '-test-' + str(cfg.dataset['testing_set_size']) + '/' + cfg.directory['testing'] + '/' + folder
	#directory = cfg.directory['testing'] + '/' + folder
	if not os.path.exists(directory):
		os.makedirs(directory)	
	
	with open( directory + '/' + links[i].split('/')[5] + '.jpg', 'wb') as handler:
		handler.write(image)
		