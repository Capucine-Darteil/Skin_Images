import pandas as pd
import numpy as np
from skimage import io
from pathlib import Path
import cv2

# Exemple chemin  : '/home/auguste/code/Capucine-Darteil/Skin_Images/raw_data/HAM10000_images_part_1'

def get_data(chemin):

    image_dir_path = '.'
    paths1 = [path.parts[-1:] for path in
         Path('chemin').rglob('*.jpg')]

    data = {}
    for i in range(len(paths1)):
        data[paths1[i][0][:-4]] = io.imread(f'chemin/{paths1[i][0]}')


    return data


def resize_data(data, size):
    resized_data = {}
    for key, image in data.items():
        resized_image = cv2.resize(image, (size,size))
        resized_data[key] = resized_image

    return resized_data
