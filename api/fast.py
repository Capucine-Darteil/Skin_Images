import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Skin_Project.ml_logic.registry import load_model

app = FastAPI()

@app.get("/")
def root():
    return {
    'greeting': 'Hello'
}
