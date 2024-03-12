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
from starlette.responses import Response
import cv2
import tensorflow as tf
from pydantic import BaseModel
import pickle

class Item(BaseModel):
    sex: str
    age: int
    localization: str

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

@app.post('/binary_classification')
async def custom_binary_classification(img: UploadFile=File(...)):

    contents = await img.read()
    image = np.fromstring(contents, np.uint8)
    image=cv2.imdecode(image, cv2.IMREAD_COLOR)


    binary_model = load_best_model()

    image_resized = cv2.resize(image, (64,64))

    threshold = THRESHOLD

    df_new_image =image_resized/255
    df_new_image = np.array(df_new_image).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    prediction = binary_model.predict(df_new_image)

    if prediction[0][0] < float(threshold) :
        result=('Not dangerous')
    else:
        result=('Yuck! You better go check that')

    return Response(content=result)

@app.post('/multiclass_classification')
async def custom_multiclass_predict(img: UploadFile=File(...)):

    contents = await img.read()
    image = np.fromstring(contents, np.uint8)
    image=cv2.imdecode(image, cv2.IMREAD_COLOR)

    local_best_model_path = f"{CHEMIN_CAT}/best_model.h5"
    multiclass_model = tf.keras.models.load_model(local_best_model_path)

    multi_image_resized = cv2.resize(image, (128,128))

    multi_new_image =multi_image_resized/255
    multi_new_image = np.array(multi_new_image).reshape(1, 128,128, 3)
    multi_prediction = multiclass_model.predict(multi_new_image)
    cat_pred = np.argmax(multi_prediction[0])

    multiclass_dict = {4:'Nævus mélanocytaire', 6:'Mélanome', 2:'Kératose séborrhéique', 1:'Carcinome basocellulaire', 0:'Kératose actinique', 5:'Lésion vasculaire', 3:'Dermatofibrome'}
    mole_type = multiclass_dict[cat_pred]

    return Response(content=mole_type)

data1 = {
    "sex":"male",
    "age":20,
    "localization":"neck"
}


@app.post('/predict_metadata')
async def custom_predict_metadata(data=data1,img: UploadFile=File(...)):
    print('test')

    with open('preproc.pkl', 'rb') as file:
        load_preproc = pickle.load(file)

    # contents = await img.read()
    # image = np.fromstring(contents, np.uint8)
    # image=cv2.imdecode(image, cv2.IMREAD_COLOR)

    # age = item.age
    # sex = item.sex
    # localization = item.localization

    # # Create the dict
    # data = {
    #     'age': [age],
    #     'sex': [sex.lower()],
    #     'localization': [localization.lower()]
    # }

    # # Create the pandas DataFrame
    # df_X = pd.DataFrame(data)
    # new_x = load_preproc.transform(df_X)

    # return Response(new_x)
    return "hello"

'''
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
'''
