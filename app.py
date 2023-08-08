from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
app = FastAPI()

MODEL_PATH ='D:/cv/abi poc cv/training/model_inception.h5'

# Load your trained model
model = load_model(MODEL_PATH)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/')
async def root():
    return {'hello': 'world'}

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    print(preds)
    preds=np.argmax(preds, axis=1)
    if preds==1:
        preds="moon"
    elif preds==2:
        preds="nonmoon"
    
    
    
    return preds

@app.post('/predict')
def upload(file: UploadFile):

    upload_dir = os.path.join(os.getcwd(), "uploads")
    # Create the upload directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # get the destination path
    dest = os.path.join(upload_dir, file.filename)
    print(dest)
    with open(dest, "wb") as buffer:
        buffer.write(file.file.read())
   # Make prediction
    preds = model_predict(dest, model)
    result=preds
    return result 