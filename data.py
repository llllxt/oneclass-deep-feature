from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
# dataset
import os
import glob
from utils.preprocess_img_coral import jpg_image_to_array
from os.path import isfile, join, isdir, exists
import random
import pandas as pd

RESCALE = 256
CHANNELS = 3
CLASSES = 2
subfd_threecls='oneclass'

def get_normal(root_dir,split,valid_set=False):
    labels = pd.read_csv(os.path.join(root_dir,split+'Labels.csv'),delimiter=',')
   
    print('checking if .ny file exists')

    x_name = join(root_dir, subfd_threecls, "{}_{}_x.npy".format(RESCALE, split))
    print(x_name)

    if exists(x_name):
        imgs = np.load(x_name)
    else:
        print('.p]npy files dont exist. Processing...')
        imgs = []
        print("sampling data...")
        folder_name = 'prepBG_'+split
        img_path = os.path.join(root_dir,folder_name)
        files_name = glob.glob(os.path.join(img_path,"*.jpeg"))
       
        if split == 'train':
            files_name = random.sample(files_name,6000)
        else:
            files_name = random.sample(files_name,5000)
        for name in files_name:
            full_name = name.split('/')[-1]
            img_name = full_name.split('.')[0]
            if labels[labels['image']==img_name]['level'].values[0] == 0:
                im_arr = jpg_image_to_array(name,sz=RESCALE)
                imgs.append(im_arr)
            else:
                continue
        print('Saving')
        np.save(x_name,imgs)
        print('normal data shape')
        print(len(imgs))
    return imgs
def get_abnormal(root_dir,split,valid_set=False):
    labels = pd.read_csv(os.path.join(root_dir,split+'Labels.csv'),delimiter=',')
    print('checking if .ny file exists')

    x_name = join(root_dir, subfd_threecls, "{}_abnormal_x.npy".format(RESCALE, split))

    if exists(x_name):
        imgs = np.load(x_name)
    else:
        print('.npy files dont exist. Processing...')
        imgs = []
        print("sampling data...")
        folder_name = 'prepBG_'+split
        img_path = os.path.join(root_dir,folder_name)
        files_name = glob.glob(os.path.join(img_path,"*.jpeg"))
        if split == 'test':
            files_name = random.sample(files_name,5000)
        for name in files_name:
            full_name = name.split('/')[-1]
            img_name = full_name.split('.')[0]
            if labels[labels['image']==img_name]['level'].values[0] == 4:
                im_arr = jpg_image_to_array(name,sz=RESCALE)
                imgs.append(im_arr)
            else:
                continue
        print('Saving')
        np.save(x_name,imgs)
        print('abnormal data shape')
        print(len(imgs))
    return imgs

def kyocera_data(data_path):

    #make reference data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    #learning data
    x_train_s, x_test_s, x_test_b = [], [], []
    x_ref, y_ref = [], []

    x_train_shape = x_train.shape
    count = 0
    for i in range(len(x_train)):
        temp = x_train[i]
        x_ref.append(temp.reshape((x_train_shape[1:])))
        y_ref.append(y_train[i])

    x_ref = np.array(x_ref)
    print(x_ref.shape)

    #6000 randomly extracted from ref data
    number = np.random.choice(np.arange(0,x_ref.shape[0]),6000,replace=False)

    x, y = [], []

    x_ref_shape = x_ref.shape

    test = []
    for i in number:
        temp = x_ref[i]
        x.append(temp.reshape((x_ref_shape[1:])))
        y.append(y_ref[i])

    # root_dir='/home/ubuntu/00_astar/00_baseline/00_drkaggle'
    root_dir = '/home/students/student3_15/00_astar/00_baseline/00_drkaggle'
    
    x_train_s = get_normal(root_dir,'train')
    x_ref = np.array(x)
    print(set(y))
    y_ref = to_categorical(y)
    print(y_ref.shape)

    #test data
    x_test_s = get_normal(root_dir,'test')
    x_test_b = get_abnormal(root_dir,'test')

    
    #resize data
    
    X_train_s = default_loader(x_train_s)
    X_ref = resize_data(x_ref)
    y_ref = np.array(y_ref)

    X_test_s = default_loader(x_test_s)
    X_test_b = default_loader(x_test_b)
    # X_train_s : normal data
    # X_ref : reference data
    # X_test_s : test normal data
    # X_test_b : test abnormal 

    return X_train_s, X_ref, y_ref, X_test_s, X_test_b


def default_loader(x):
    x_out = []
    for i in range(len(x)):
        img = cv2.resize(x[i],(224,224))
        x_out.append(img.astype('float32') / 255)

    return np.array(x_out)
def resize_data(x):
    x_out = []
    for i in range(len(x)):
        img = cv2.cvtColor(x[i], cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img,(224,224))
        x_out.append(img.astype('float32') / 255)

    return np.array(x_out)
