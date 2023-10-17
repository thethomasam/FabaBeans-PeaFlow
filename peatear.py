#!/usr/bin/env python3

import click
import cv2
import numpy as np
import os
import pandas as pd
import sys
# import tensorflow as tf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# from peatrain import confusion_matrix_heatmap

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Compose, ToTensor


from math import ceil
from random import randint

from cnn_trainer import NeuralNetwork
SQUARE_SIZE = 64


@click.command()
@click.argument("image_path")
@click.argument("out_dir_path")
@click.argument("rows", type=int)
@click.argument("columns", type=int)
@click.argument("transformer", type=int)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose mode.")
@click.version_option("1.0.0", message="%(version)s")
def main(image_path: str, out_dir_path: str, rows: int, columns: int, verbose: bool,transformer: bool,isResnet: bool,):
    peatearer(image_path,out_dir_path,rows=4, cols=4,transformer=False,verbose=False)
    """Quick and easy data generation for ranked set sampling.
    
    Splits the image given by IMAGE_PATH into ROWS rows and COLUMNS columns.
    Outputs everything to the directory given by OUT_DIR_PATH.

    """
def peatearer(image_path,out_dir_path,rows=5, columns=4,verbose=False,transformer=False,isResnet=False):
    if not os.path.isfile(image_path):
        print(
            "\033[91mError: Given `image_path` doesn't exist.\033[00m", file=sys.stderr
        )
        sys.exit(1)

    out_dir_path = os.path.dirname(out_dir_path)
    os.makedirs(out_dir_path, exist_ok=True)
    if isResnet:
        model_name='RESNET_nnmodel.pth'
    elif transformer:
        model_name='transformer_nnmodel.pth'
    else:
        model_name='cnnmodel.pth'
    # nn_model = tf.keras.models.load_model(os.path.join("models", "fababeans-64.h5"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load('./'+model_name))
    model.eval()
   
    
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    data = {"row": [], "column": [], "samples": []}

    for i in range(rows):
        for j in range(columns):
            c1 = (height // rows * i, width // columns * j)
            c2 = (height // rows * (i + 1), width // columns * (j + 1))

            cell_img = img[c1[0] : c2[0], c1[1] : c2[1]]
            output_copy = cell_img.copy()
            cv2.imwrite('output_copy.jpg',output_copy)
            

            cell_height = cell_img.shape[0]
            cell_width = cell_img.shape[1]

            num_sub_rows = cell_height // SQUARE_SIZE * 2 - 1
            num_sub_columns = cell_width // SQUARE_SIZE * 2 - 1

            x_origin = (cell_width - (ceil(num_sub_columns / 2) * SQUARE_SIZE)) // 2
            y_origin = (cell_height - (ceil(num_sub_rows / 2) * SQUARE_SIZE)) // 2

            samples = 0

            for m in range(num_sub_rows):
                for n in range(num_sub_columns):
                    x1, y1 = (
                        x_origin + (n * (SQUARE_SIZE // 2)),
                        y_origin + (m * (SQUARE_SIZE // 2)),
                    )
                    x2, y2 = (x1 + SQUARE_SIZE, y1 + SQUARE_SIZE)


                    subcell_img = cell_img[y1:y2+2, x1:x2+2]
                   
                    image = cv2.cvtColor(subcell_img, cv2.COLOR_BGR2RGB)
                    transform = Compose([ToTensor()])  # Normalises to 0-1
                    image = transform(image).unsqueeze(dim=0).to(device)
                    prediction =nn.functional.softmax(model(image)[0], dim=0)
                   
                    

                    if not prediction[0] > prediction[1]:
                        cv2.circle(
                            output_copy,
                            (x1 + SQUARE_SIZE // 2, y1 + SQUARE_SIZE // 2),
                            32,
                            (0, 255, 0),
                            1,
                        )
                        samples += 1
                    start_point = (x2, y1)
                    end_point = (x2, y2)
                    cv2.line(img, start_point, end_point, (255, 0, 0), 5)
                    
                
            # if i < rows - 1:
            #     start_point = (0, c1[0])
            # end_point = (width, c1[0])
            # cv2.line(img, start_point, end_point, (0, 0, 255), 5)


            if j <= columns :  # Draw vertical lines between columns
                start_point = (c1[1], 0)
                end_point = (c1[1], height)
                cv2.line(img, start_point, end_point, (0, 255, 0), 5)




            data["row"].append(i + 1)
            data["column"].append(j + 1)
            data["samples"].append(samples)

            cv2.imwrite(
                os.path.join(out_dir_path, "{},{}.jpg".format(i + 1, j + 1)),
                output_copy,
            )
    cv2.imwrite('line_image.jpg',img)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(out_dir_path, "ranking.csv"), index=False)


if __name__ == "__main__":
    main()
