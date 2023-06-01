datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, fill_mode='nearest')

dir_It = datagen.flow_from_directory(
    "Data - All Trying/train_normal",
    batch_size=1,
    save_to_dir="Data - All Trying/train_preprocessed/normal",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))

##


datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

dir_It = datagen.flow_from_directory(
    "Data - All Trying/train_normal",
    batch_size=1,
    save_to_dir="Data - All Trying/train_preprocessed/normal",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))

from tensorflow import keras

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from PIL import Image

import numpy as np
import os
%matplotlib inline

import seaborn as sns
import keras.utils as image
import cv2

datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, fill_mode='nearest')

dir_It = datagen.flow_from_directory(
    "Data - All Trying/train_adeno",
    batch_size=1,
    save_to_dir="Data - All Trying/train_preprocessed/adenocarcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))

##


datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

dir_It = datagen.flow_from_directory(
    "Data - All Trying/train_adeno",
    batch_size=1,
    save_to_dir="Data - All Trying/train_preprocessed/adenocarcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))
    
##

datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, fill_mode='nearest')

dir_It = datagen.flow_from_directory(
    "Data - All Trying/train_large",
    batch_size=1,
    save_to_dir="Data - All Trying/train_preprocessed/large.cell.carcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))

##


datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

dir_It = datagen.flow_from_directory(
    "Data - All Trying/train_large",
    batch_size=1,
    save_to_dir="Data - All Trying/train_preprocessed/large.cell.carcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))

datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, fill_mode='nearest')

dir_It = datagen.flow_from_directory(
    "Data - All Trying/train_small",
    batch_size=1,
    save_to_dir="Data - All Trying/train_preprocessed/squamous.cell.carcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))

##


datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

dir_It = datagen.flow_from_directory(
    "Data - All Trying/train_small",
    batch_size=1,
    save_to_dir="Data - All Trying/train_preprocessed/squamous.cell.carcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))

datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, fill_mode='nearest')

dir_It = datagen.flow_from_directory(
    "Data - All Trying/valid_normal",
    batch_size=1,
    save_to_dir="Data - All Trying/valid_preprocessed/normal",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))

##


datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

dir_It = datagen.flow_from_directory(
    "Data - All Trying/valid_normal",
    batch_size=1,
    save_to_dir="Data - All Trying/valid_preprocessed/normal",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))
    
##
    

    
    
    
    
datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, fill_mode='nearest')

dir_It = datagen.flow_from_directory(
    "Data - All Trying/valid_adeno",
    batch_size=1,
    save_to_dir="Data - All Trying/valid_preprocessed/adenocarcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))

##


datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

dir_It = datagen.flow_from_directory(
    "Data - All Trying/valid_adeno",
    batch_size=1,
    save_to_dir="Data - All Trying/valid_preprocessed/adenocarcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))
    
##


    


    
    
datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, fill_mode='nearest')

dir_It = datagen.flow_from_directory(
    "Data - All Trying/valid_large",
    batch_size=1,
    save_to_dir="Data - All Trying/valid_preprocessed/large.cell.carcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))

##


datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

dir_It = datagen.flow_from_directory(
    "Data - All Trying/valid_large",
    batch_size=1,
    save_to_dir="Data - All Trying/valid_preprocessed/large.cell.carcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))
    
##

    
    

datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, fill_mode='nearest')

dir_It = datagen.flow_from_directory(
    "Data - All Trying/valid_small",
    batch_size=1,
    save_to_dir="Data - All Trying/valid_preprocessed/squamous.cell.carcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))

##


datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

dir_It = datagen.flow_from_directory(
    "Data - All Trying/valid_small",
    batch_size=1,
    save_to_dir="Data - All Trying/valid_preprocessed/squamous.cell.carcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))
    
##

datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, fill_mode='nearest')

dir_It = datagen.flow_from_directory(
    "Data - All Trying/test_normal",
    batch_size=1,
    save_to_dir="Data - All Trying/test_preprocessed/normal",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))

##


datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

dir_It = datagen.flow_from_directory(
    "Data - All Trying/test_normal",
    batch_size=1,
    save_to_dir="Data - All Trying/test_preprocessed/normal",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))
    
##

    

    
    
    
    
datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, fill_mode='nearest')

dir_It = datagen.flow_from_directory(
    "Data - All Trying/test_adeno",
    batch_size=1,
    save_to_dir="Data - All Trying/test_preprocessed/adenocarcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))

##


datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

dir_It = datagen.flow_from_directory(
    "Data - All Trying/test_adeno",
    batch_size=1,
    save_to_dir="Data - All Trying/test_preprocessed/adenocarcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))
    
##


    
    
datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, fill_mode='nearest')

dir_It = datagen.flow_from_directory(
    "Data - All Trying/test_large",
    batch_size=1,
    save_to_dir="Data - All Trying/test_preprocessed/large.cell.carcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))

##


datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

dir_It = datagen.flow_from_directory(
    "Data - All Trying/test_large",
    batch_size=1,
    save_to_dir="Data - All Trying/test_preprocessed/large.cell.carcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))
    
##
    
    

datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, fill_mode='nearest')

dir_It = datagen.flow_from_directory(
    "Data - All Trying/test_small",
    batch_size=1,
    save_to_dir="Data - All Trying/test_preprocessed/squamous.cell.carcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))

##


datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

dir_It = datagen.flow_from_directory(
    "Data - All Trying/test_small",
    batch_size=1,
    save_to_dir="Data - All Trying/test_preprocessed/squamous.cell.carcinoma",
    save_prefix="",
    save_format='png',
)
for _ in range(len(dir_It)):
    img, label = dir_It.next()
    plt.imshow((img[0] * 255).astype(np.uint8))
    
##
