#!/usr/bin/python3

# Copyright 2019 SAGI and the University of Adelaide
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""PEAUTILITY.PY (limit 72 chars for doc, PEP8)
TODO: some documentation here. basically we use this file to store all of
the 'utility' functions that are used by each of the scripts (pealabel.py,
peascription.py, peatrain.py) to eliminate code duplication.

Associated test suites should appear in peautility_test.py once I get
around to it.

"""

import os
import sys
import time





def get_annotated_filename(image_filename, label, x1, y1, x2, y2):
    """Return the annotated filename for the given image.

    Arguments:  
    image_filename -- a string containing the filename for an image  
    label -- an integer specifying the label (0 or 1) for the image  
    x1 -- an integer for the x-coordinate of the top-left corner of the
          bounding box  
    y1 -- an integer for the y-coordinate of the top-left corner of the
          bounding box  
    x2 -- an integer for the x-coordinate of the bottom-right corner of
          the bounding box  
    y2 -- an integer for the y-coordinate of the bottom-right corner of
          the bounding box

    Returns:
    a string containing the image filename, embellished with the label and
    bounding box information.

    Usage:
    get_annotated_filename("images/testimage001.jpg", 1, 23, 14, 65, 55)  
        #>>> "testimage001-1-23,14-65,55.jpg"

    After annotating, the output image filename will be of the form  
        filename-label-x1,y1-x2,y2.jpg  
    where filename is the name of the image (not including any directory
    structure), label denotes the positive/negative image designation 
    (0 for negative, 1 for positive) and (x1,y1) and (x2,y2) are the 
    coordinates for the upper-left and bottom-right of the bounding box 
    respectively.
    """
    parts = os.path.splitext(os.path.basename(image_filename))
    filename = parts[0]
    extension = parts[1]
    filename += ("-" + str(label))
    filename += ("-" + str(x1) + "," + str(y1))
    filename += ("-" + str(x2) + "," + str(y2))
    filename += extension

    return filename




def get_base_filename(image_filename):
    """Return the base filename for the given image. 
    
    Arguments:  
    image_filename -- a string containing the filename for an image

    Returns:  
    a string containing the image filename, with all pealabel embellishments
    (file extension, directory, label/bounding box information) stripped. 

    Usage:  
    get_base_filename("images/testimage001-1-23,14-65,55.jpg")    
        #>>> "testimage001"
    """
    truncated = os.path.splitext(image_filename)[0]
    truncated = os.path.split(truncated)[-1]
    parts = truncated.split("-")
    return parts[0]


def get_bounding_box(image_filename):
    """Return the bounding box information for the given image.

    Arguments:  
    image_filename -- a string containing the filename for an image

    Returns:  
    a list of integers, containing the bounding box information for the image.

    Usage:  
    get_bounding_box("images/testimage001-1-23,14-65,55.jpg")  
        #>>> [23, 14, 65, 55]
    """
    truncated = os.path.splitext(image_filename)[0]
    parts = truncated.split("-")
    top_left = parts[-2].split(",")
    bot_right = parts[-1].split(",")

    return list(map(int, top_left + bot_right))


def get_label(image_filename):
    """Return the label information for the given image.

    Arguments:  
    image_filename -- a string containing the filename for an image

    Returns:  
    the integer specifying the annotation label for the image (a 0 specifies
    a negative label, and a 1 specifies a positive label.

    Usage:  
    get_label("images/testimage001-1-23,14-65,55.jpg")  
        #>>> 1
    """
    truncated = os.path.splitext(image_filename)[0]
    parts = truncated.split("-")

    return int(parts[-3])


def log(message):
    """Log a message to standard output (e.g. in verbose mode).
    
    Arguments:  
    message -- a string containing the message to be logged

    Usage:  
    log("This message is written to stdout.")  
        #>>> This message is written to stdout.
    """
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def print_and_wait(message, wait):
    """Print a message to standard output and wait for a given
    number of seconds.

    Arguments:
    message -- a string containing the message to be printed  
    wait -- a number (integer or float) specifying the number of
            seconds to wait after printing the message

    Usage:
    print_and_wait("Print this message and wait two seconds.\n", 2)
        #>>> Print this message and wait two seconds.
    """
    sys.stdout.write(message)
    sys.stdout.flush()
    time.sleep(wait)


def print_intro_animation():
    """Prints a whimsical animation to the standard output stream."""
    # These characters do not appear properly on some operating
    # systems. Not a critical issue since this animation is just for
    # fun anyway, but noted here for the sake of completeness.
    print_and_wait("(•_•)", 0.5)
    print_and_wait("\r( •_•)>⌐■-■", 0.5)
    print_and_wait("\r(⌐■_■)     ", 1)
    print_and_wait('\r(⌐■_■) "Let\'s do this."', 1)
    sys.stdout.write("\n")
    sys.stdout.flush()

