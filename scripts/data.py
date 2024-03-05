import pandas as pd
import numpy as np
from skimage import io
from pathlib import Path
import cv2
import os

SIZE = os.environ.get('SIZE')

# Exemple chemin  : '/home/auguste/code/Capucine-Darteil/Skin_Images/raw_data/HAM10000_images_part_1'

def get_data(chemin):

    image_dir_path = '.'
    paths1 = [path.parts[-1:] for path in
         Path('chemin').rglob('*.jpg')]

    data = {}
    for i in range(len(paths1)):
        data[paths1[i][0][:-4]] = io.imread(f'chemin/{paths1[i][0]}')
    return data

# Concatener les dictionnaires d'images. Un dictionnaire par dossier avec des images.
def merge_dicts(dict_1, dict_2):
    images_dict = dict_1.upload(dict_2)
    return images_dict

# Modifier la taille des images
def resize_data(data):
    resized_data = {}
    for key, image in data.items():
        resized_image = cv2.resize(image, (SIZE,SIZE))
        resized_data[key] = resized_image

    return resized_data

# Reshape data et transformer en DataFrame:
def flat_images(data):
    flat_data = {}
    for key, image in data.items():
        flat_image = image.flatten()
        flat_data[key] = flat_image

    df_images = pd.DataFrame(flat_data).transpose()
    return df_images

def df_final(chemin, df_images):
    metadf = pd.read_csv(chemin)
    metadf = metadf.set_index('image_id')
    df = pd.concat([metadf,df_images],axis=1)
    return df
