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
from werkzeug.utils import secure_filename
from fastapi.responses import JSONResponse

import shutil
import requests
import json
import logging
import time
import datetime
import tensorflow as tf

app = FastAPI()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s'
    )
logger=logging.getLogger()

MODEL_PATH ='model_inception.h5'

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


async def get_uploaded_file(file):
    logger.info("File Uploading ...")
    try:
        file_raw = secure_filename(file.filename)
        filename = "out_images/added_images/"+file_raw

        with open(filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"filename: {filename}")

    except Exception as e:
        print(e)
        filename == ''
        logger.error("No file selected for uploading.")
        resp = JSONResponse(content={'error_msg': 'No file selected for uploading'})
        resp.status_code = 200
        return resp
    


    return filename, file_raw
    
def opencv_op_2(filename):
    img = tf.keras.utils.load_img(filename, target_size=[224,224])
    img = np.array(img)/255
    img = np.expand_dims(img, axis=0)

    return img
    
def model_predict(img):
    # print(img_path)
    # img = image.load_img(img_path, target_size=(224, 224))

    # # Preprocessing the image
    # x = image.img_to_array(img)
    # # x = np.true_divide(x, 255)
    # ## Scaling
    # x=x/255
    # x = np.expand_dims(x, axis=0)
    is_moon=None

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(img)
    logger.info("moon classifictaion results ==> {}".format(preds))
    preds=np.argmax(preds, axis=1)
    if preds==1:
        is_moon="Yes"
    elif preds==2:
        is_moon="No"
      
    
    
    return is_moon

@app.post('/predict')
async def upload(file: UploadFile):
    start = time.time()
    filename_ = await get_uploaded_file(file)
    print(filename_)
    filename = filename_[0]
    ts = int(time.time())
    file_raw_ = filename_[1]
    file_raw_split = file_raw_.split(".")
    file_raw_name = file_raw_split[0]
    file_raw_ext = file_raw_split[1]
    file_raw = file_raw_name+'_'+str(ts)+'.'+file_raw_ext
    error_msg = None
    is_moon= None
    # upload_dir = os.path.join(os.getcwd(), "uploads")
    # # Create the upload directory if it doesn't exist
    # if not os.path.exists(upload_dir):
        # os.makedirs(upload_dir)

    # # get the destination path
    # dest = os.path.join(upload_dir, file.filename)
    # print(dest)
    # with open(dest, "wb") as buffer:
        # buffer.write(file.file.read())
   # # Make prediction
    preds = model_predict(img=opencv_op_2(filename))
    resp = JSONResponse(
            {
            "is_moon": preds
            })  
    end = time.time()
    logger.info("Time taken to upload file is {}".format(end - start))
    resp.status_code = 200
    newfilename = "out_images/added_images/" + file_raw
    shutil.copyfile(filename, newfilename)
    os.remove(filename)
    return resp 
