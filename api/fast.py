import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from Skin_Project.ml_logic.registry import load_model
from PIL import Image
import os
from Skin_Project.ml_logic.data import get_data, resize_data, flat_images
from Skin_Project.params import *
from Skin_Project.ml_logic.registry import load_best_model

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# app.state.model = load_model()

@app.get("/")
def root():
    return {
    'Test': 'This is not a test... LOL'
}

@app.post("/image")
def image_process(CHEMIN_TEST,gender,age,location):
    image = get_data(CHEMIN_TEST)
    image_resized = resize_data(image,int(IMAGE_SIZE))
    new_image_processed = flat_images(image_resized)
    new_image_processed = new_image_processed.reset_index()
    new_image_processed = new_image_processed.drop(columns='index')
    metadata_dict = {'Age':age,'Sex':gender, 'Localization':location}
    image_metadata = pd.DataFrame(metadata_dict,index=[0])
    df_new_image = pd.concat([image_metadata,new_image_processed],axis=1)

    return df_new_image

def custom_binary_predict(df_new_image,THRESHOLD):
    model = load_best_model()
    threshold = THRESHOLD
    df_new_image = df_new_image.drop(columns=['Age', 'Sex', 'Localization'])
    df_new_image = df_new_image/255
    df_new_image = np.array(df_new_image).reshape(len(df_new_image), IMAGE_SIZE, IMAGE_SIZE, 3)
    prediction = model.predict(df_new_image)
    print(prediction)
    if prediction[0][0] < float(threshold) :
        return 0
    else:
        return 1

def custom_multiclass_predict(df_new_image):
    model = load_best_model()
    df_new_image = df_new_image.drop(columns=['Age', 'Sex', 'Localization'])
    df_new_image = df_new_image/255
    df_new_image = np.array(df_new_image).reshape(len(df_new_image), IMAGE_SIZE, IMAGE_SIZE, 3)
    prediction = model.predict(df_new_image)
    cat_pred = np.argmax(prediction[0])
    multiclass_dict = {4:'Nævus mélanocytaire', 6:'Mélanome', 2:'Kératose séborrhéique', 1:'Carcinome basocellulaire', 0:'Kératose actinique', 5:'Lésion vasculaire', 3:'Dermatofibrome'}
    multiclass_dict[cat_pred]
    return cat_pred
