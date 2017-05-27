import os
import json
import csv
from PIL import Image

import config as cfg
import random

json_file_name = "mentis-amneville-one.json"
csv_file_name = 'occupancy117.csv'

with open(json_file_name, "r") as f:
    data = json.loads(f.read())
    
with open(csv_file_name, 'rb') as csvfile:

    readCSV=csv.reader(csvfile, delimiter=',')
    labels=[]
    date_times=[]
    slot_ids=[]
    
    for row in readCSV:
        label=row[0]
        date_time=row[1]
        slot_id=row[2]
        
        labels.append(int(label))
        date_times.append(date_time[0:4] + date_time[5:7] + date_time[8:10] + date_time[11:13] + date_time[14:16] + date_time[17:19])
        slot_ids.append(slot_id)
        
print("Number of rows read from %s file:" % csv_file_name)
print(len(labels))
#print(len(date_times))
#print(len(slot_ids))

data_set = range( cfg.dataset['size'] )

random.shuffle(data_set)

train_set = data_set[:(int( cfg.dataset['train_ratio'] * len(data_set))) ]
test_set = data_set[(int( cfg.dataset['train_ratio'] * len(data_set))): ]

#print "train indices:", train_set
#print "test indices:", test_set

if not os.path.exists("locarno_cropped"):
    os.makedirs("locarno_cropped/training/label0")
    os.makedirs("locarno_cropped/training/label1")
    os.makedirs("locarno_cropped/testing/label0")
    os.makedirs("locarno_cropped/testing/label1")
    
#print(data['spots']['11700']['external_id'])

iterator = 0
labelled_crops = 0
# for spot in data['spots']:
    # print(spot)
    # iterator += 1
    # dic_str = os.path.join("locarno_cropped", data['spots'][str(spot)]['external_id'])
    # if not os.path.exists(dic_str):
        # os.makedirs(dic_str)

dates = sorted(os.listdir("117"))
for date in dates:
    imagefiles = sorted(os.listdir(os.path.join("117", date)))
    for imagefile in imagefiles:
        orig_img = Image.open(os.path.join("117", date, imagefile))
        for spot in data['spots']:
            
            if iterator < cfg.dataset['size']:
                
                if iterator in train_set:
                    train_or_test = cfg.directory['training']
                else:
                    train_or_test = cfg.directory['testing']
                    
                date_time = imagefile[0:14]
                slot = str(data['spots'][str(spot)]["external_id"])
                print("Iteration: %d, Slot external id: %s" % (iterator, slot))
                
                #print(len(date_times))
                for i in range(len(date_times)):
                    # print(date_times[i])
                    # print(date_time)
                    # print(slot_ids[i])
                    # print(slot)
                    if date_times[i] == date_time and slot_ids[i] == slot:
                        
                        if labels[i] == 0:
                            label = cfg.dataset['label0_folder']
                        else:
                            label = cfg.dataset['label1_folder']

                        print("Label found in labels file: %s" % (label))
                        labelled_crops += 1
                        print("Number of labelled crops: %d" % (labelled_crops))
                        img = orig_img.rotate(data['spots'][str(spot)]['rotation'], expand=True)
                        w, h = img.size
                        left = int(data['spots'][str(spot)]['roi_rel']['x']*w)
                        top = int(data['spots'][str(spot)]['roi_rel']['y']*h)
                        right = int(data['spots'][str(spot)]['roi_rel']['width']*w) + left
                        bottom = int(data['spots'][str(spot)]['roi_rel']['height']*h) + top
                        img = img.crop((left, top, right, bottom))
                        path = os.path.join( train_or_test, label, slot + '_' + imagefile)
                        #print(path)
                
                        img.save(path)
                
                
            iterator += 1
