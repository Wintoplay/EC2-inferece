import torch
import torchvision.transforms as T
import pandas as pd
import glob
import boto3
import matplotlib.pyplot as plt
import json
import os
import sys
from PIL import Image
import numpy as np
import shutil
import math

# Load model in jit
model = torch.jit.load(r"C:\Users\winro\Desktop\Homework\Bigdata\mushroom project\model\model_cpu.pt")

# Load lightning model
#model = load_from_checkpoint()

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
s3 = boto3.resource('s3',aws_access_key_id="ASIARP5DJRVXNGOTHGHO",
                    aws_secret_access_key="6Momvoe1DsAxWWz+88Avx+eszEyQobe8GQo6WMVh",
                    aws_session_token="FwoGZXIvYXdzEL///////////wEaDEyc4+jGdu8+u72C8yLHAZMvTl28N4O9rknCt0n7shyYn69kSc5LGdaO495MJe3as27b9u2qD4VItxx0dEa1fO7obi3P0GEfm8CVmj42WcwvpLIzjb9HN9Dt845u0H0h25ti9r8u0XUunqwS1t9/sB35N1CcFz1ZtI0wirKUnB1sdxVEmbjLZ83dKjd5myfRqVcvHPlHtXQcoLLtIA08ZLIuO6WOfLtMYWsenFXBEW2/nOmWoZ8h1Bk8nXU9TbVxBqB4pcUNSzb1BSovJKg50/5EjoC7PTQo7tv+qQYyLTfb+CblvcNZdLA28GN8ofS7gCcYHm9EN7F6+aTR6yeeUqf4GIq+igPwvqNksw==")
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
poisonous_count = 0
edible_count = 0
for image in files_list:
    rgb_image = Image.open(temp_path+image).convert('RGB')
    transformed_image = transforms(rgb_image)
    (transformed_image.shape)
    prediction = model(torch.unsqueeze(transformed_image, 0)).detach().numpy()
    prediction = np.argmax(prediction)
    prediction_result.append(prediction)
    file_name.append(image)
    if (prediction==1):
        poisonous_count += 1
    else:
        edible_count += 1
# dictionary of lists 
result_dict = {'Name': file_name, 'Poisonous': prediction_result}
print(result_dict)

# Pandas write cvs
df = pd.DataFrame(result_dict)
df.to_csv("./result.csv",index=False)

# Print analytics
print("poisonous_count: ", str(poisonous_count))
print("edible_count: ", str(edible_count))

# Testing of the model is only done in sagemaker for GPU utilization

# Visualize Images
images = glob.glob(temp_path)
image_count = len(images)
max_image_show = 20
n_col = 4
if image_count<max_image_show:
    n_row = math.ceil(image_count/n_col)
else:
    n_row = math.ceil(max_image_show/n_col)
n_row
_, axs = plt.subplots(n_row, n_col, figsize=(4*n_col, 4*n_row))
axs = axs.flatten()
for img, ax in zip(images, axs):
    ax.imshow(Image.open(img))
    ax.axis("off")
plt.show()

# Clean temp dir
shutil.rmtree(temp_path)