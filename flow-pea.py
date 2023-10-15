import os
import click
import cv2
import os
import pandas as pd
import sys


from glob import glob
from random import randint
import subprocess

from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch


 
from peascription import peascripter
# from pea_trainer_pytorch import train_network
# from peatear import peatearer
from pealabel import peaLabler
from peadivide import peatearer

"""
Automated Plant Analysis Workflow

This script automates the process of plant analysis using a combination of image classification models. The workflow includes steps for image preprocessing, model training, analysis, and labeling.

Parameters:
- peascription_in_dir_path: Input directory path for raw drone-acquired images.
- peascription_out_dir_path: Output directory path for processed images and annotations.
- trainner_annotations: Path to the CSV file containing annotations for training the model.
- model_output: Path to the directory where trained models will be saved.
- pea_tearer_image: Path to an image for analysis using the peatearer module.
- pea_tearer_out: Output directory path for analyzed images.
- rows, cols: Number of rows and columns for image analysis grid.
- epochs: Number of epochs for training the model.
- transformer: Boolean indicating whether the transformer model should be used.
- isResnet: Boolean indicating whether the ResNet model should be used.
- verbose: Boolean indicating whether verbose mode is enabled.

Functions:
- delete_files_recursively(directory_path): Deletes files recursively in the specified directory.
- all_runner(...): Executes the entire workflow, including preprocessing, training, analysis, and labeling.

Usage:
Run this script with the specified parameters to perform automated plant analysis. The script guides you through options to run training, analysis, and labeling steps interactively.

Note:
- You can choose to use either the transformer model or ResNet for training.
- The script is designed to automate various stages of plant analysis using different modules and models.
- It is recommended to modify the default parameters based on your use case and directory structure.

"""


verbose=False

peascription_in_dir_path = "./images/peascription-in/"
peascription_out_dir_path = "./images/peascription-out/" 

trainner_annotations='./images/peascription-out/annotations.csv'
model_output="./models"



pea_tearer_image="./analysis/15-6-10-orig.jpg"
pea_tearer_out ="./output/"
rows=5
cols=8
pealabler_image='./images/peascription-in/15-6-1-orig.jpg'

isResnet=False

_Transformer=False


# model switches
if _Transformer:
    _EPOCHS=3
    from transformer_train import train_network

elif isResnet:
    from cnn_trainer import train_network
    _EPOCHS=20
else:
    print('cnn_trainer')
    _EPOCHS=20
    from cnn_trainer import train_network



# cols=3
# rows=2
def delete_files_recursively(directory_path):
    try:
        for root, dirs, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                os.remove(file_path)
        print(f"Deleted Extras")
    except Exception as e:
        print(f"Error: {e}")

# Provide the path to the directory where you want to delete files



def all_runner(peascription_in_dir_path,peascription_out_dir_path,trainner_annotations,model_output,pea_tearer_image, pea_tearer_out,rows, cols,epochs,transformer,isResnet,verbose=False):
    # delete_files_recursively(peascription_out_dir_path)
    # peascripter(peascription_in_dir_path,peascription_out_dir_path,verbose=False)
    # print('Done Pea Scription')
    # run_trainer=input('Do you want to run the trainer Y/N\n')
    # if run_trainer=='Y' or run_trainer=='y':
    #     train_network(trainner_annotations,model_output,isResnet=isResnet,transformer=transformer,epochs=epochs)
    
    peatearer(pea_tearer_image, pea_tearer_out, rows, cols, transformer=transformer,isResnet=isResnet,verbose=False)

    # run_labler = input('Do you want to run the labler Y/N\n')
    # if run_labler=='Y' or run_labler=='y':
    #     peaLabler(pea_tearer_out, peascription_out_dir_path, verbose)
    #     re_train = input('Do you want to retrain the model Y/N\n')
    #     if re_train=='Y' or re_train=='y':
    #         train_network(trainner_annotations,model_output,isResnet=isResnet)
    #         re_analyse_path=input('Do you want to retrain the model Y/N\n')
    #         if re_analyse_path:
    #             peatearer(pea_tearer_image, pea_tearer_out, rows, cols, verbose=False)
    

    


if __name__ == "__main__":
    all_runner(peascription_in_dir_path=peascription_in_dir_path,peascription_out_dir_path=peascription_out_dir_path,trainner_annotations=trainner_annotations,model_output=model_output,pea_tearer_image=pea_tearer_image, pea_tearer_out=pea_tearer_out,rows=rows,cols=cols,epochs=_EPOCHS,transformer=_Transformer,isResnet=isResnet)