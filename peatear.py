#!/usr/bin/env python3

import click
import cv2
import numpy as np
import os
import pandas as pd
import sys


from cnn_trainer import NeuralNetwork
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Compose, ToTensor

from math import ceil
from math import sqrt
from math import floor
from random import randint


SQUARE_SIZE = 64

# copy/pasted from new peatear.py version: should work as is.
def merge_overlapping_circles(circles):
    """Merge the overlapping circles in the given list.

    Arguments:
    circles -- the zipped list of (circle_centre, circle_radius) circles

    Returns: a zipped list of (circle_centre, circle_radius) circles
             such that no circles overlap each other (the overlapping
             circles are merged into larger circles).
    """

    # Use (squared) Euclidean distance to measure between circles in 2D
    def squared_dist(x, y):
        return (x[0] - y[0])**2 + (x[1] - y[1])**2

    # Find indices of overlapping circles
    def find_overlap_indices():
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                # Overlap when distance between centres <= sum of radii
                if (squared_dist(circles[i][0], circles[j][0])
                        < (circles[i][1] + circles[j][1])**2):
                    return (i, j)
        # If the function got to the end, no overlap was found.
        return None

    # Find two overlapping circles
    overlap_indices = find_overlap_indices()
    if (overlap_indices is not None):
        i, j = overlap_indices

        # Merge the i,j indexed circles and recurse.
        # The merged circle has a centre given by the midpoint of the
        # two circles, and a diameter that spans both circles.
        new_circle_centrex = (circles[i][0][0] + circles[j][0][0]) // 2
        new_circle_centrey = (circles[i][0][1] + circles[j][0][1]) // 2
        new_circle_centre = (new_circle_centrex, new_circle_centrey)
        dist = sqrt(squared_dist(circles[i][0], circles[j][0]))
        dist = int(floor(dist))
        new_circle_radius = (circles[i][1] + circles[j][1] + dist) // 2

        circles[i] = (new_circle_centre, new_circle_radius)
        circles = circles[:j] + circles[(j + 1):]
        return merge_overlapping_circles(circles)
    else:
        # No more overlaps. Done.
        return circles


@click.command()
@click.argument("image_path")
@click.argument("out_dir_path")
@click.argument("rows", type=int)
@click.argument("columns", type=int)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose mode.")
@click.version_option("1.0.0", message="%(version)s")
def cli(image_path: str, out_dir_path: str, rows: int, columns: int, verbose: bool):
    peatearer(image_path, out_dir_path, rows, columns, verbose,transformer=False,isResnet=False)
    """Quick and easy data generation for ranked set sampling.
    
    Splits the image given by IMAGE_PATH into ROWS rows and COLUMNS columns.
    Outputs everything to the directory given by OUT_DIR_PATH.

    """
def peatearer(image_path, out_dir_path, rows, columns, verbose,transformer=False,isResnet=False):
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
    print(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load('./'+model_name))
    model.eval()
    # nn_model = tf.keras.models.load_model(os.path.join("models", "pnn-cnn-peascription.h5"))

    img = cv2.imread(image_path)
    height, width, _ = img.shape

    data = {"row": [], "column": [], "samples": []}

    for i in range(rows):
        for j in range(columns):
            c1 = (height // rows * i, width // columns * j)
            c2 = (height // rows * (i + 1), width // columns * (j + 1))

            cell_img = img[c1[0] : c2[0], c1[1] : c2[1]]
            output_copy = cell_img.copy()

            cell_height = cell_img.shape[0]
            cell_width = cell_img.shape[1]

            num_sub_rows = cell_height // SQUARE_SIZE * 2 - 1
            num_sub_columns = cell_width // SQUARE_SIZE * 2 - 1

            x_origin = (cell_width - (ceil(num_sub_columns / 2) * SQUARE_SIZE)) // 2
            y_origin = (cell_height - (ceil(num_sub_rows / 2) * SQUARE_SIZE)) // 2

            samples = 0
            circle_centres = []

            for m in range(num_sub_rows):
                for n in range(num_sub_columns):
                    x1, y1 = (
                        x_origin + (n * (SQUARE_SIZE // 2)),
                        y_origin + (m * (SQUARE_SIZE // 2)),
                    )
                    x2, y2 = (x1 + SQUARE_SIZE, y1 + SQUARE_SIZE)

                    subcell_img = cell_img[y1:y2, x1:x2] / 255.0

                    subcell_img = cell_img[y1:y2+2, x1:x2+2]
                   
                    subcell_img = cv2.cvtColor(subcell_img, cv2.COLOR_BGR2RGB)
                    transform = Compose([ToTensor()])  # Normalises to 0-1
                    subcell_img = transform(subcell_img).unsqueeze(dim=0).to(device)
                    prediction =nn.functional.softmax(model(subcell_img)[0], dim=0)

                    if not prediction[0].item() >  prediction[1].item():
                        cv2.circle(
                            output_copy,
                            (x1 + SQUARE_SIZE // 2, y1 + SQUARE_SIZE // 2),
                            32,
                            (0, 255, 0),
                            1,
                        )
                        centre = (x1 + SQUARE_SIZE // 2, y1 + SQUARE_SIZE // 2)
                        circle_centres.append(centre)

            circle_radii = [32]*len(circle_centres)
            circles = list(zip(circle_centres, circle_radii))

            # Merge circles
            circles = merge_overlapping_circles(circles)

            # Draw the circles and count the samples here instead.
            for circle in circles:
                samples += 1

                centre = circle[0]
                radius = circle[1]
                colour = (255, 85, 255)  # magenta
                thickness = 8  # very thick circle perimeter
                cv2.circle(output_copy, centre, radius, colour, thickness)

            data["row"].append(i + 1)
            data["column"].append(j + 1)
            data["samples"].append(samples)
            print(samples)
            cv2.imwrite(
                os.path.join(out_dir_path, "{},{}.jpg".format(i + 1, j + 1)),
                output_copy,
            )

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(out_dir_path, "ranking.csv"), index=False)


if __name__ == "__main__":
    cli()