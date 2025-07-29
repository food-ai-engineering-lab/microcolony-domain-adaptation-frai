import os
import numpy as np
import random
from shutil import copyfile

## Select data group based on experimental conditions
# group = '20x-3h'
# group = '20x-5h'
# group = '60x-3h'
group = '60x-3h-2'
ECgroup = 'Ec-'+group

# Define paths for the source dataset and the train/test directories
root = '/data2/microcolony-data/'+group+'/'
classes = os.listdir(root)
trainval = '/data2/microcolony-data/train/'+group+'/'
test = '/data2/microcolony-data/test/'+group+'/'

## Treat all Ec strains as a single class
root_Ec = os.path.join(root, ECgroup)
if not os.path.exists(root_Ec):
    os.makedirs(root_Ec)
    classes = os.listdir(root) # update classes
if len(os.listdir(root_Ec)) == 0: # run this code only if the Ec folder is empty
    for i in range(len(classes)):
        if 'Ec' in classes[i] and classes[i] != ECgroup:
            root_cls = os.path.join(root, classes[i])
            ds = os.listdir(root_cls)
            for img in ds:
                copyfile(root_cls+'/'+img, root_Ec+'/'+img)
                
## For imbalanced dataset - find the minimum data size
datasize = {}
for i in range(len(classes)):
    if 'Ec' not in classes[i] or classes[i] == ECgroup: # pass strain-specific Ec folders
        root_cls = os.path.join(root, classes[i])
        datasize[classes[i]] = len(os.listdir(root_cls))
print(datasize)
min_datasize = min(datasize.values())

''' 
Rename files with typos    
'''           
img_path = '/data2/microcolony-data/20x-3h/Ec-20x-3h/' # change this folder
imgs = list(sorted(os.listdir(img_path)))

for j in range(len(imgs)):
    file = os.path.join(img_path, imgs[j])
    old_name = file.split('/')[-1]
    head, body = old_name.split('-20x-3h-')
    date = body[:6]
    if date == '22016-': # change this string with the ones with typos
        date = '220106' # change this string with the desired names
        tail = body [5:]
        new_name = head + '-20x-3h-' + date + tail
        os.rename(img_path + old_name, img_path + new_name)
        
''' 
Train-val/test split   
'''
for i in range(len(classes)): # for each species
    if 'Ec' not in classes[i] or classes[i] == ECgroup: # pass strain-specific Ec folders
        root_cls = os.path.join(root, classes[i])
        trainval_cls = os.path.join(trainval, classes[i])
        ds = os.listdir(root_cls)
        
        ## Initialize empty lists
        test_list = []
        trainval_list = []
        
        ## Define dateset split ration and data size
        test_p = .15
        test_size = int(np.floor(min_datasize * test_p))
        trainval_size = int(min_datasize-int(np.floor(min_datasize * test_p)))
        
        # ## Get the data collection dates
        # dates = []
        # for img in ds:
        #     date = img.split(group+'-')[-1]
        #     date = date[:6]
        #     if date not in dates:
        #         dates.append(date)
        # print(classes[i], dates) # Check if there's any typo and run the cell above if needed
        
        # ## Split data pool based on the data collection dates
        # n_testdates = max(1, int(len(dates)*0.5))
        # random.shuffle(dates)
        # test_dates = dates[:n_testdates]
        # trainval_dates = dates[n_testdates:]
        # print(classes[i])
        # for img in ds:
        #     date = img.split(group+'-')[-1][:6]
        #     if date in test_dates:
        #       test_list.append(img)
        #     else:
        #         trainval_list.append(img)
     
        # ## Random sampling (if we use data collection dates)
        # test_set = random.sample(test_list, test_size)
        # trainval_set = random.sample(trainval_list, trainval_size)
        
        ## Random sampling 
        test_set = random.sample(ds, test_size)
        for img in ds:
            if img not in test_set:
                trainval_list.append(img)
        trainval_set = random.sample(trainval_list, trainval_size)
                
        ## Make sub-folders for each class if not exist
        if not os.path.exists(trainval_cls):
            os.makedirs(trainval_cls)
        if not os.path.exists(test):
            os.makedirs(test)
            
        ## Copy files
        for img in trainval_set:
            copyfile(root_cls+'/'+img, trainval_cls+'/'+img)
        for img in test_set:
            copyfile(root_cls+'/'+img, test+'/'+img)
