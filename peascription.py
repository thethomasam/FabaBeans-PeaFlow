#!/usr/bin/env python3

import click
import cv2
import os
import pandas as pd
import sys

from glob import glob
from random import randint


SQUARE_SIZE = 64


@click.command()
@click.argument("in_dir_path")
@click.argument("out_dir_path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose mode.")
@click.version_option("1.0.0", message="%(version)s")
def main(in_dir_path: str, out_dir_path: str, verbose: bool):
    peascripter(in_dir_path, out_dir_path, verbose=False)




def peascripter(in_dir_path, out_dir_path, verbose=False):
   
#    for command line interface reading options
    """Yet another incredible script by Lachlan.
    
    I'm too lazy to write usage instructions this time. Just give 'er a whirl
    and hope for the best. 60% of the time, it works every time.
    
    """

    if not os.path.isdir(in_dir_path):
        print(
            "\033[91mError\033[00m: Given `in_dir_path` doesn't exist.", file=sys.stderr
        )
        sys.exit(1)
    print(out_dir_path)
    out_dir_path = os.path.dirname(out_dir_path)
    os.makedirs(out_dir_path, exist_ok=True)
    os.makedirs(os.path.join(out_dir_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir_path, "images", "plant"), exist_ok=True)
    os.makedirs(os.path.join(out_dir_path, "images", "notplant"), exist_ok=True)

    data = {"path": [], "label": []}

    for orig_img_path in glob(os.path.join(in_dir_path, "*-orig.jpg")):
        dot_img_path = orig_img_path.replace("orig", "dot")

        orig_img = cv2.imread(orig_img_path)
        dot_img = cv2.imread(dot_img_path)

        height, width, _ = orig_img.shape

        dot_img_hsv = cv2.cvtColor(dot_img, cv2.COLOR_BGR2HSV_FULL)
        #dot_img_blurred = cv2.GaussianBlur(dot_img_hsv, (15, 15), 0)
        dot_img_blurred = dot_img_hsv

        
        # Use upper/lower masking to isolate red HSV region
        lower_mask = cv2.inRange(dot_img_blurred, (0, 70, 200), (5, 255, 255))
        upper_mask = cv2.inRange(dot_img_blurred, (160, 70, 200), (255, 255, 255))

        

        dot_img_thresh = cv2.bitwise_or(lower_mask, upper_mask)
        #dot_img_thresh = cv2.inRange(dot_img_blurred, (160, 150, 0), (255, 255, 255))

       
       

        candidate_dot_contours, _ = cv2.findContours(
            dot_img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        test_dot_images=dot_img_thresh
        test_con=cv2.drawContours(test_dot_images, candidate_dot_contours, -1, (0, 255, 0), 3)

        # cv2.imwrite('./dot_img_thresh1.jpg',dot_img_thresh)
        # cv2.imwrite('./dot_img_thresh.jpg',test_con)
        dot_contours = [c for c in candidate_dot_contours if cv2.contourArea(c) >= 5.0]

        used_squares = []

        for c in dot_contours:
            M = cv2.moments(c)
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])

            half_square_size = SQUARE_SIZE // 2

            nudge = half_square_size // 4
            x += randint(-nudge, nudge)
            y += randint(-nudge, nudge)

            if x < half_square_size:
                x = half_square_size
            if x > width - half_square_size:
                x = width - half_square_size
            if y < half_square_size:
                y = half_square_size
            if y > height - half_square_size:
                y = height - half_square_size

            x1, y1 = (x - half_square_size, y - half_square_size)
            x2, y2 = (x + half_square_size, y + half_square_size)

            square_img = orig_img[y1:y2, x1:x2]

            square_img_path = os.path.join(
                "images",
                "plant",
                "{}-{}-{},{}-{},{}.jpg".format(
                    os.path.splitext(os.path.basename(orig_img_path))[0],
                    1,
                    x1,
                    y1,
                    x2,
                    y2,
                ),
            )
            joined_path=os.path.join(out_dir_path, square_img_path)
            cv2.imwrite(joined_path, square_img)
            used_squares.append(((x1 - 20, y1 - 20), (x2 + 20, y2 + 20)))

            data["path"].append(joined_path)
            data["label"].append(1)

        for _ in range(len(used_squares) * 2):
            half_square_size = SQUARE_SIZE // 2

            while True:
                x = randint(half_square_size, width - half_square_size)
                y = randint(half_square_size, height - half_square_size)

                x1, y1 = (x - half_square_size, y - half_square_size)
                x2, y2 = (x + half_square_size, y + half_square_size)

                found_overlap = False
                for us in used_squares:
                    if (
                        x1 < us[1][0]
                        and x2 > us[0][0]
                        and y1 < us[1][1]
                        and y2 > us[0][1]
                    ):
                        found_overlap = True
                        break
                if found_overlap:
                    continue

                square_img = orig_img[y1:y2, x1:x2]

                square_img_path = os.path.join(
                    "images",
                    "notplant",
                    "{}-{}-{},{}-{},{}.jpg".format(
                        os.path.splitext(os.path.basename(orig_img_path))[0],
                        0,
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                )
                joined_path=os.path.join(out_dir_path, square_img_path)
                cv2.imwrite(joined_path,square_img)
                used_squares.append(((x1, y1), (x2, y2)))

                data["path"].append(joined_path)
                data["label"].append(0)
                break

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(out_dir_path, "annotations.csv"), index=False)


if __name__ == "__main__":
    main()
