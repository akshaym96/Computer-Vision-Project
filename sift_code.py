import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

print(cv2.__version__)
sift  = cv2.xfeatures2d.SIFT_create()

def get_dense_descriptors(data):
    temp_des = []
    des = np.empty(shape=[0,128])
    for i in data:
        kp = [cv2.KeyPoint(y, x,scale) for y in range(0, i.shape[0], step) 
                                for x in range(0, i.shape[1], step)]
        kp, dense_feat = sift.compute(i, kp)
        dense_feat = np.array(dense_feat)
        temp_des.append(dense_feat)
        des = np.append(des,dense_feat,axis=0)
    return temp_des, des 

# for filename in os.path.join():

img = cv2.imread("training_set/2.jpg")
# print(img)
# sift_dict = {}
# for filename in os.listdir('training_set/'):
#     img = cv2.imread('training_set/'+filename)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('img',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     break
    
    

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img,None)

print(len(kp1))



#CSV file reading

photo_to_bus_dict = {}


with open('train_photo_to_biz_ids.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            photo_to_bus_dict[row[0]] = row[1]
#             print(f'\t{row[0]} works in the {row[1]} department')
            line_count += 1
    print(f'Processed {line_count} lines.')
print(len(photo_to_bus_dict))
    
bus_to_labels_dict = {}    
    
with open('train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            bus_to_labels_dict[row[0]] = [row[1]]
#             print(f'\t{row[0]} works in the {row[1]} department')
            line_count += 1
    print(f'Processed {line_count} lines.')  
print(len(bus_to_labels_dict))

hist_label = [0]*9

for filename in os.listdir('training_set'):
    photo_id = filename.split('.')
    lst_labels = bus_to_labels_dict[photo_to_bus_dict[photo_id[0]]]
#     print(lst_labels)
    labels = lst_labels[0].split(' ')
    if len(labels)!=0:
        for i in labels:
            if i is not '':
                hist_label[int(i)]+=1
                

# print(bus_to_labels_dict[photo_to_bus_dict['204149']])
print("Histogram distribution for the labels in the training_set")
for i in hist_label:
    print(i)
    
print(hist_label)
