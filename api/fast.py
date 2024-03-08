import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from Skin_Project.ml_logic.registry import load_model
from PIL import Image
import os
from Skin_Project.ml_logic.data import get_data, resize_data, flat_images
from Skin_Project.params import *

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
    metadata_dict = {'Age':age,'Sex':gender, 'Localization':location}
    image_metadata = pd.DataFrame(metadata_dict)
    new_image_processed = flat_images()
    #récupérer chemin de la photo
    #preprocess l'image
    #model.predict image (besoin d'avoir un load model fonctionnel)
    pass
