#import google.cloud.storage as gcs
from google.cloud import storage as gcs
import csv
import requests
import urllib2
import httplib

import config as cfg
import tensorflow as tf
import os
import random

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

data_set = range( cfg.dataset['size'] )

random.shuffle(data_set)

train_set = data_set[:(int( cfg.dataset['train_ratio'] * len(data_set))) ]
test_set = data_set[(int( cfg.dataset['train_ratio'] * len(data_set))): ]

print "train indices:", train_set
print "test indices:", test_set

training_set_size = str( int(cfg.dataset['size'] * cfg.dataset['train_ratio']) )
testing_set_size = str( int(cfg.dataset['size'] * cfg.dataset['test_ratio']) )

httplib.HTTPConnection._http_vsn = 10
httplib.HTTPConnection._http_vsn_str = 'HTTP/1.0'

#downloading images for training
for i in range(cfg.dataset['size']):

  print i
  folder = cfg.dataset['label' + str(labels[i]) + '_folder']
  
  request = urllib2.Request(links[i])
  try:
    response = urllib2.urlopen(request)
    image = response.read()
  except httplib.IncompleteRead, e:
    image = e.partial
  httplib.HTTPConnection._http_vsn = 11
  httplib.HTTPConnection._http_vsn_str = 'HTTP/1.1'
  
  if i in train_set:
    train_or_test = cfg.directory['training']
  else:
    train_or_test = cfg.directory['testing']
  
  directory = train_or_test + '/' + folder
  if not os.path.exists(directory):
    os.makedirs(directory)
  
  with open( directory +  '/' + links[i].split('/')[5] + '.jpg', 'wb') as handler:
    handler.write(image)
