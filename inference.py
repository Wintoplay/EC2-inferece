import torch
import torchvision.transforms as T
import pandas as pd

import glob

import boto3

import json
import os
import sys
from PIL import Image
import numpy as np
import shutil

# Load model in jit
model = torch.jit.load("/home/admin/efs/model/model_cpu.pt")

# Prepare Transformation
transforms =T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
])

# Make temp dir if not isExist
temp_path = "./temp/"
isExist = os.path.exists(temp_path)
if not isExist:
    os.mkdir(temp_path)

# Download all images from the specific directory in the specified S3 bucket
s3 = boto3.resource('s3')
def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
download_s3_folder(bucket_name = "mushroomprojectbigdata", s3_folder = "Inference", local_dir = "temp")

# Predict
prediction_result = []
file_name = []
files_list = os.listdir(temp_path)
for image in files_list:
    rgb_image = Image.open(temp_path+image).convert('RGB')
    transformed_image = transforms(rgb_image)
    (transformed_image.shape)
    prediction = model(torch.unsqueeze(transformed_image, 0)).detach().numpy()
    prediction = np.argmax(prediction)
    prediction_result.append(prediction)
    file_name.append(image)

# dictionary of lists 
result_dict = {'Name': file_name, 'Poisonous': prediction_result}
print(result_dict)

# Pandas write cvs
df = pd.DataFrame(result_dict)
df.to_csv("./result.csv",index=False)

# Clean temp dir
shutil.rmtree(temp_path)