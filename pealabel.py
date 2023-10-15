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

"""PEALABEL.PY constructs a Graphical User Interface for manually
annotating images for the PEATRAIN.PY training dataset.

This script may be called from the command line:  
    ./pealabel.py [OPTIONS] IN_DIRECTORY OUT_DIRECTORY

This script may also be imported and run from the Python3 interpreter:  
    import pealabel  
    pealabel.run(in_directory, out_directory, verbose)

When run, PEALABEL.PY generates a Graphical User Interface that
successively views images from the IN_DIRECTORY. For each image, the 
user clicks to place boxes to be cropped and sorted into a training 
dataset in the OUT_DIRECTORY. The image crops are sorted into two 
categories:  
    - LEFT CLICKING designates a NEGATIVE to the cropped image
      (i.e. the Machine Learning algorithm should use the image as
      a negative example),  
    - RIGHT CLICKING designates a POSITIVE to the cropped image
      (i.e. the Machine Learning algorithm should use the image as
      a positive example).  
The user can also press BACKSPACE to undo annotations. Pressing the
SPACEBAR moves to the next image, if there is one. Pressing ESC ends 
the annotating.

Once the sorting has completed, the cropped images in the OUT_DIRECTORY
are labelled with their designation (0 for negative, 1 for positive) 
and bounding box coordinates. For convenience, this label information 
is also written to the output file ANNOTATIONS.CSV in a separate 
column.

When running from the command line or the Python3 interpreter, the user
can set a flag for additional output:  
    -v, --verbose    Enable verbose mode.

Alternatively, the associated Jupyter Notebook PEALABEL.IPYNB imports
these functions and provides the same functionality, but embedded in 
an informative, tutorial-style interface.  
TODO: Jupyter Notebook pending.
"""

import glob
import os
import sys

# Import common utility methods from PEAUTILITY.PY
import peautility as pu

# If this script is called at the command line, we use the Command Line
# Interface Creation Kit (CLICK)
if (__name__ == "__main__"):
    import click

# Using OpenCV and the pandas Python Data Analysis libraries
import cv2
import numpy as np
import pandas as pd
import random


# The size (in pixels) for the cropped output images
_OUTPUT_IMAGE_SIZE = 64

# The filename for the output CSV file containing the annotation data
_OUTPUT_CSV = "annotations.csv"

# OpenCV uses BGR (Blue-Green-Red) colours
_COLOUR = {"RED": (0, 0, 255),
           "GREEN": (0, 255, 0),
           "BLUE": (255, 0, 0),
           "WHITE": (255, 255, 255),
           "BLACK": (0, 0, 0)}

# The OpenCV GUI polls for keyboard presses: these are the codes.
# (Tested on Windows, Mac OSX and (Debian) Linux, 22/10/2019.)
# We check these last 8 bits to determine what key was pressed.
_KEY = {"BACKSPACE": 0x08,
        "MACOSX_DELETE": 0x7f,
        "ESC": 0x1b,
        "SPACEBAR": 0x20}

# Initial parameters for the drawn GUI window
_WINDOW_NAME = "Pinder"
_WINDOW_WIDTH = 800
_WINDOW_HEIGHT = 800

# A Trackbar implements zooming functionality on the photos.
# Default zoom value is 50%.
_TRACKBAR_ZOOM_NAME = "Zoom %"
_TRACKBAR_ZOOM_MINVAL = 0
_TRACKBAR_ZOOM_MAXVAL = 100
_TRACKBAR_ZOOM_VAL = 50

# A Trackbar allows panning left/right on the (zoomed) photos.
# Default value starts in the top-left corner.
_TRACKBAR_LEFTRIGHT_NAME = "Left/right"
_TRACKBAR_LEFTRIGHT_MINVAL = 0
_TRACKBAR_LEFTRIGHT_MAXVAL = 200
_TRACKBAR_LEFTRIGHT_VAL = 0

# A Trackbar allows panning up/down on the (zoomed) photos.
# Default value starts in the top-left corner.
_TRACKBAR_UPDOWN_NAME = "Up/down"
_TRACKBAR_UPDOWN_MINVAL = 0
_TRACKBAR_UPDOWN_MAXVAL = 200
_TRACKBAR_UPDOWN_VAL = 0


def gui(data, verbose=False):
    """Construct the 'Pinder' window to view and annotate an image.

    Arguments:  
    data -- a mutable dictionary that will log the images annotated
            by the user in the GUI operation  
    verbose -- if True, print log/debug information to stdout  
               (default False)

    This gui() function comprises the brunt of the pealabel.py 
    annotation code. Calling this function instantiates the image 
    viewing window, implements the zooming and left/right/up/down 
    panning functionality with the trackbar widgets, and implements 
    annotation of the images through mouse clicks by the user.

    This function is called by the run() function during normal 
    pealabel.py operation; there should be no need to call this 
    function directly.
    """
    # Shuffle the images in the input directory to mitigate any bias.
    #TODO: .jpg images only for the moment. We might want to extend
    #      here so that pealabel.py can handle other image file types
    #      (e.g. png image files)?
    images = glob.glob(os.path.join(_IN_DIRECTORY, "*.jpg"))
    random.shuffle(images)
    if (verbose):
        pu.log("Loaded images from " + _IN_DIRECTORY + ", shuffled:")
        [pu.log(image) for image in images]
        
    for image in images:
        # Implement a simple UNDO functionality by keeping track of
        # the coloured annotation squares the user adds during this
        # run of the program.
        actions = []
        
        # Instantiate the GUI window
        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(_WINDOW_NAME, _WINDOW_WIDTH, _WINDOW_HEIGHT)
        if (verbose):
            message = "Drawing window " + _WINDOW_NAME
            message += " with width=" + str(_WINDOW_WIDTH)
            message += ", height=" + str(_WINDOW_HEIGHT)
            pu.log(message)
        
        image_full = cv2.imread(image)
        image_xmin = 0
        image_xmax = len(image_full[0])
        image_ymin = 0
        image_ymax = len(image_full)

        # The image view is deduced robustly using the coordinates
        # of the top-left corner (view_x, view_y). This view is
        # completely determined by the values on the panning trackbars,
        # and the zoom fraction controlled by the zoom trackbar.
        global _TRACKBAR_ZOOM_VAL
        global _TRACKBAR_LEFTRIGHT_VAL
        global _TRACKBAR_UPDOWN_VAL
        x_fraction = _TRACKBAR_LEFTRIGHT_VAL / _TRACKBAR_LEFTRIGHT_MAXVAL
        view_x = image_xmin + int(x_fraction*image_xmax)
        y_fraction = _TRACKBAR_UPDOWN_VAL / _TRACKBAR_UPDOWN_MAXVAL
        view_y = image_ymin + int(y_fraction*image_ymax)
        zoom_fraction = _TRACKBAR_ZOOM_VAL / _TRACKBAR_ZOOM_MAXVAL

        
        def update_view():
            """Update the current image view based upon the specified 
            zoom and left/right/up/down panning trackbars, and draw 
            all of the coloured annotation boxes that are visible in 
            that view.
            """
            # Make a clean copy of the image upon each update.
            # (This copying is necessary to clear the coloured boxes
            # when the view shifts.)
            image_copy = image_full.copy()

            # Update the view based upon the values of the zoom and
            # panning trackbars
            nonlocal x_fraction
            nonlocal y_fraction
            nonlocal zoom_fraction
            
            nonlocal view_x
            nonlocal image_xmin
            nonlocal image_xmax
            view_x = image_xmin + int(x_fraction*image_xmax)            
            view_xmin = view_x
            view_xmax = view_xmin + image_xmax - int(zoom_fraction*image_xmax)
            
            nonlocal view_y
            nonlocal image_ymin
            nonlocal image_ymax
            view_y = image_ymin + int(y_fraction*image_ymax)
            view_ymin = view_y
            view_ymax = view_ymin + image_ymax - int(zoom_fraction*image_ymax)
            
            # A square view looks best for the images, so truncate as
            # necessary to make the view square.
            x_size = view_xmax - view_xmin
            y_size = view_ymax - view_ymin
            if (x_size < y_size):
                view_ymax = view_ymin + x_size
            elif (y_size < x_size):
                view_xmax = view_xmin + y_size

            # Correct for when the view window is out of bounds with
            # respect to the full image
            if (view_xmax > image_xmax):
                over_x = view_xmax - image_xmax
                view_xmin -= over_x
                view_xmax -= over_x
                view_x = view_xmin

            if (view_ymax > image_ymax):
                over_y = view_ymax - image_ymax
                view_ymin -= over_y
                view_ymax -= over_y
                view_y = view_ymin
                 
            # Redraw all of the green/red bounding boxes
            if (verbose):
                pu.log("Redrawing bounding boxes in current image view.")
                
            nonlocal data
            for output_image in data["path"]:
                # Only display the coloured boxes for this current image
                if (pu.get_base_filename(image)
                        == pu.get_base_filename(output_image)):
                    x1, y1, x2, y2 = pu.get_bounding_box(output_image)
                    label = pu.get_label(output_image)
                
                    # Draw a coloured bounding box if any part of it
                    # is visible in the current image view.
                    box_visible = (((x1 > view_xmin and x1 < view_xmax)
                                    and (y1 > view_ymin and y1 < view_ymax))
                                   or ((x2 > view_xmin and x2 < view_xmax)
                                    and (y2 > view_ymin and y2 < view_ymax)))
                    if (box_visible):
                        if (label == 0):
                            box_colour = _COLOUR["RED"]
                        else:
                            box_colour = _COLOUR["GREEN"]
                        if (verbose):
                            message = "Bounding box (" + str(x1) + ", "
                            message += str(y1) +  ")-(" + str(x2) + ", "
                            message += str(y2) + ") is visible: drawing "
                            if (label == 0):
                                message += "RED bounding box."
                            else:
                                message += "GREEN bounding box."
                            pu.log(message)
                        cv2.rectangle(image_copy, (x1, y1), (x2, y2),
                                      box_colour, thickness=4)

            view = image_copy[view_ymin:view_ymax, view_xmin:view_xmax]
            cv2.imshow(_WINDOW_NAME, view)

            
        # Implement the Zoom trackbar
        def on_trackbar_zoom(val):
            """Callback function for the Zoom trackbar. This function 
            is called whenever the user changes the value on the Zoom 
            trackbar, and adjusts the zoom fraction for the image 
            view accordingly.

            Arguments:  
            val -- an integer representing the current value of the  
                   zoom trackbar (between 0 and 100)
            """
            if (verbose):
                pu.log("Registered Trackbar(Zoom) input: " + str(val))

            # Update the trackbar value
            global _TRACKBAR_ZOOM_VAL
            _TRACKBAR_ZOOM_VAL = val

            # Update the zoom fraction
            nonlocal zoom_fraction
            zoom_fraction =  _TRACKBAR_ZOOM_VAL / _TRACKBAR_ZOOM_MAXVAL

            # If the zoom is 100%, we try zoom=99.50% instead. This
            # zoom value, slightly less than 100%, is arguably a more
            # useful zoom value, so that we're not just zooming in on
            # a single pixel.
            if (zoom_fraction == 1.0):
                zoom_fraction = 0.9950

            # Update the image view to reflect the new zoom.
            update_view()

            
        # Initialise the Zoom trackbar (start at 50% zoom)
        _TRACKBAR_ZOOM_VAL = 50
        cv2.createTrackbar(_TRACKBAR_ZOOM_NAME, _WINDOW_NAME,
                           _TRACKBAR_ZOOM_VAL, _TRACKBAR_ZOOM_MAXVAL,
                           on_trackbar_zoom)

        
        # Implement the Left/Right panning trackbar
        def on_trackbar_leftright(val):
            """Callback function for the Left/Right panning trackbar.
            This function is called whenever the user changes the value 
            on the Left/Right panning trackbar, and adjusts the position 
            of the image view accordingly (as a 'fraction' of how far 
            it can move rightward).

            Arguments:  
            val -- an integer representing the current value of the 
                   left/right panning trackbar (between 0 and some 
                   maximum value, determined by the global variable 
                   _TRACKBAR_LEFTRIGHT_MAXVAL. More gradations allow 
                   for smoother, but 'slower', panning.)
            """
            if (verbose):
                pu.log("Registered Trackbar(Left/Right) input: " + str(val))

            # Update the trackbar value
            global _TRACKBAR_LEFTRIGHT_VAL
            _TRACKBAR_LEFTRIGHT_VAL = val

            # Update the left/right panning fraction
            nonlocal x_fraction
            x_fraction = (_TRACKBAR_LEFTRIGHT_VAL
                          / _TRACKBAR_LEFTRIGHT_MAXVAL)
            
            # Update the image view to reflect the new position.
            update_view()

            
        # Initialise the Left/Right panning trackbar (start in the
        # top left corner)
        _TRACKBAR_LEFTRIGHT_VAL = 0
        cv2.createTrackbar(_TRACKBAR_LEFTRIGHT_NAME, _WINDOW_NAME,
                           _TRACKBAR_LEFTRIGHT_VAL,
                           _TRACKBAR_LEFTRIGHT_MAXVAL, on_trackbar_leftright)

        
        # Implement the Up/Down panning trackbar 
        def on_trackbar_updown(val):
            """Callback function for the Up/Down panning trackbar. This
            function is called whenever the user changes the value on 
            the Up/Down panning trackbar, and adjusts the position of 
            the image view accordingly (as a 'fraction' of how far it 
            can move downward).

            Arguments:  
            val -- an integer representing the current value of the 
                   up/down panning trackbar (between 0 and some maximum
                   value, determined by the global variable 
                   _TRACKBAR_UPDOWN_MAXVAL. More gradations allow for 
                   smoother, but 'slower', panning.)
            """
            if (verbose):
                pu.log("Registered Trackbar(Up/Down) input: " + str(val))

            # Update the trackbar value
            global _TRACKBAR_UPDOWN_VAL
            _TRACKBAR_UPDOWN_VAL = val

            # Update the up/down panning fraction
            nonlocal y_fraction
            y_fraction = (_TRACKBAR_UPDOWN_VAL
                          / _TRACKBAR_UPDOWN_MAXVAL)

            # Update the image view to reflect the new position.
            update_view()
            

        # Initialise the Up/Down panning trackbar (start in the
        # top left corner)
        _TRACKBAR_UPDOWN_VAL = 0
        cv2.createTrackbar(_TRACKBAR_UPDOWN_NAME, _WINDOW_NAME,
                           _TRACKBAR_UPDOWN_VAL,
                           _TRACKBAR_UPDOWN_MAXVAL, on_trackbar_updown)

        
        # Implement the left-click/right-click functionality
        def on_mouse(ev, x, y, flags, param):
            """Callback function for the mouse events. This function 
            is called whenever a mouse movement/click is detected on 
            the image view. We use this function to implement the image
            annotations based on left-click and right-click input 
            from the user.

            Arguments:  
            ev -- an integer representing the captured mouse event  
            x -- the x-position of the mouse for the event  
            y -- the y-position of the mouse for the event  
            flags -- additional flags (not used)  
            param -- additional params (not used)  
            """
            # The x- and y-coordinates of the mouse are relative to
            # the current image view window. To accurately crop the
            # image and draw the coloured boxes, We also require
            # the absolute coordinates.
            nonlocal view_x
            nonlocal view_y
            abs_x = view_x + x
            abs_y = view_y + y
            
            if (verbose):
                message = "Registered mouse event: ("
                message += str(x) + ", " + str(y) + ") [Image abs: ("
                message += str(abs_x) + ", " + str(abs_y) + ")]"
                pu.log(message)
            
            # Apply a label if either mouse button was clicked.
            if (ev == cv2.EVENT_LBUTTONDOWN or ev == cv2.EVENT_RBUTTONDOWN):
                half_square_size = _OUTPUT_IMAGE_SIZE // 2
                
                if (ev == cv2.EVENT_LBUTTONDOWN):
                    if (verbose):
                        message = "Registered left-button click: ("
                        message += str(x) + ", " + str(y) + ") [Image abs: ("
                        message += str(abs_x) + ", " + str(abs_y) + ")]"
                        pu.log(message)
                    
                    # Apply a negative label on a left-button click.
                    label = 0
                else:
                    if (verbose):
                        message = "Registered right-button click: ("
                        message += str(x) + ", " + str(y) + ") [Image abs: ("
                        message += str(abs_x) + ", " + str(abs_y) + ")]"
                        pu.log(message)
                        
                    # Apply a positive label on a right-button click.
                    label = 1

                    # Also add a random 'nudge' to mitigate bias (since
                    # a user will tend to target the exact centre of a
                    # plant with their positive label click)
                    nudge = half_square_size // 2
                    abs_x += random.randint(-nudge, nudge)
                    abs_y += random.randint(-nudge, nudge)
                
                # Handle the image corners by truncating the cropped
                # image appropriately.
                nonlocal image_xmax
                nonlocal image_ymax
                abs_x = max(abs_x, half_square_size)
                abs_x = min(abs_x, image_xmax - half_square_size)
                abs_y = max(abs_y, half_square_size)
                abs_y = min(abs_y, image_ymax - half_square_size)

                # Crop the image for the output
                x1, y1 = (abs_x - half_square_size, abs_y - half_square_size)
                x2, y2 = (abs_x + half_square_size, abs_y + half_square_size)
                cropped_image = image_full[y1:y2, x1:x2]
                path = pu.get_annotated_filename(image, label, x1, y1, x2, y2)
                path = os.path.join("images", path)
                data["path"].append(path)
                cv2.imwrite(os.path.join(_OUT_DIRECTORY, path), cropped_image)

                # Keep track of this action for a possible UNDO.
                actions.append(path)

                # Update the image view to show the colour boxes.
                update_view()
                
        
        cv2.setMouseCallback(_WINDOW_NAME, on_mouse)
        
        # Finally, update the initial image view here.
        update_view()

        # An initial resize of the window makes the pealabel GUI
        # render properly on Mac OSX:
        cv2.resizeWindow(_WINDOW_NAME, _WINDOW_WIDTH, _WINDOW_HEIGHT)

        while (True):
            # The GUI polls for a key press while the window is active.
            pressed_key = cv2.waitKey(0)

            # Extract the last 8 bits to distinguish the key code
            pressed_key = pressed_key & 0xff

            if (pressed_key == _KEY["BACKSPACE"] or
                    pressed_key == _KEY["MACOSX_DELETE"]):
                if (verbose):
                    message = "Registered BACKSPACE key press: "
                    message += str(hex(pressed_key))
                    pu.log(message)
                
                # Undo the most recent action if the user pressed
                # BACKSPACE (or DELETE on a Mac).
                if (len(actions) > 0):
                    most_recent = actions.pop()

                    # Delete this image annotation from the data listing
                    data["path"].remove(most_recent)
                    
                    # Delete the cropped image from the output
                    os.remove(os.path.join(_OUT_DIRECTORY, most_recent))

                    # Update the view to reflect the change.
                    update_view()

            if (pressed_key == _KEY["SPACEBAR"]):
                if (verbose):
                    message = "Registered SPACEBAR key press: "
                    message += str(hex(pressed_key))
                    pu.log(message)
                # Go to the next image (if there is one) when the
                # SPACEBAR is pressed.
                cv2.destroyAllWindows()
                break

            if (pressed_key == _KEY["ESC"]):
                if (verbose):
                    message = "Registered ESC key press: "
                    message += str(hex(pressed_key))
                    pu.log(message)

                    pu.log("Closing all OpenCV windows.")

                # Exit the script when the ESC key is pressed.
                cv2.destroyAllWindows()
                return

    if (verbose):
        pu.log("gui() function finished.")
            

def peaLabler(in_directory, out_directory, verbose=False):
    """Runs the pealabel.py script for the image annotating.

    Arguments:  
    in_directory -- the input directory containing the images to 
                    be viewed  
    out_directory -- the output directory to put the cropped images  
    verbose -- if True, print log/debug information to stdout  
               (default False)

    This run() function sets up the input/output directories for 
    running the pealabel program, before instantiating the GUI for 
    image viewing and annotation. After control returns from the GUI, 
    this function logs the image annotations in a CSV file, 
    ANNOTATIONS.CSV, and exits.
    
    Call this function to run the pealabel program from the Python3
    interpreter:  
        import pealabel  
        pealabel.run(in_directory, out_directory, verbose)
    """
    # Print a little animation upon script start if in verbose mode.
    if (verbose):
        pu.print_intro_animation()
        
    # Set the input directory. Stop the script if the input directory
    # does not exist.
    try:
        set_input_directory(in_directory)
    except FileNotFoundError as e:
        error_message = "\033[91mERROR:\033[00m "
        error_message += "The given input directory " + e.args[0]
        error_message += " does not exist. Stopping script."
        pu.log(error_message)
        return
    if (verbose):
        pu.log("Input directory set to: " + _IN_DIRECTORY)

    # Set the output directory (and create it if it doesn't exist).
    set_output_directory(out_directory)
    if (verbose):
        pu.log("Output directory set to: " + _OUT_DIRECTORY)
    os.makedirs(os.path.join(_OUT_DIRECTORY, "images"), exist_ok=True)
    
    # Set up a data dictionary to track the annotations.
    data = {"path": [], "label": []}

    # If an ANNOTATIONS.CSV file already exists, we read it in here
    # so that any previous annotations/bounding boxes appear in the
    # image view.
    if (os.path.exists(os.path.join(_OUT_DIRECTORY, _OUTPUT_CSV))):
        read_data = pd.read_csv(os.path.join(_OUT_DIRECTORY, _OUTPUT_CSV))
        data["path"] = read_data["path"].tolist()

    # Instantiate and delegate to the GUI for the image viewing and
    # annotation.
    gui(data, verbose)

    # After the annotating is done, we construct a labels column for
    # the annotations CSV file. (It's simpler to do this here at the
    # end, rather than keeping track during the GUI run.)
    for image in data["path"]:
        data["label"].append(pu.get_label(image))
    
    # Write the output annotations to the annotations.csv file.     
    annotations = pd.DataFrame(data)
    annotations_file = os.path.join(_OUT_DIRECTORY, _OUTPUT_CSV)
    if (verbose):
        message = "Writing output annotations to: " + annotations_file
        pu.log(message)
    annotations.to_csv(os.path.join(_OUT_DIRECTORY, _OUTPUT_CSV),
                       index=False)

    if (verbose):
        pu.log("run() function finished.")
        

def set_input_directory(in_directory):
    """Set the relative input directory to in_directory. Raises a
    FileNotFoundError exception if the given directory does not exist.

    Arguments:  
    in_directory -- a string representing the input directory

    Raises:  
    FileNotFoundError -- if the given directory does not exist.

    Usage:  
    set_input_directory("./test/test_inputdirectory")
    """
    if (os.path.isdir(in_directory)):
        global _IN_DIRECTORY
        _IN_DIRECTORY = in_directory
    else:
        raise FileNotFoundError(in_directory)


def set_output_directory(out_directory):
    """Set the relative output directory to out_directory. Creates the 
    directory if it does not already exist.

    Arguments:  
    out_directory -- a string representing the new output directory

    Usage:  
    set_output_directory("./test/test_outputdirectory")
    """
    global _OUT_DIRECTORY
    _OUT_DIRECTORY = out_directory

    os.makedirs(_OUT_DIRECTORY, exist_ok=True)


# CLICK constructs the Interface when we call this pealabel.py script
# from the command line with  
#     ./pealabel.py [OPTIONS] IN_DIRECTORY OUT_DIRECTORY
if (__name__ == "__main__"):
    @click.command()
    @click.argument("in_directory")
    @click.argument("out_directory")
    @click.option("--verbose", "-v", is_flag=True,
                  help="Enable verbose mode.")
    @click.version_option("1.1.0", message="%(version)s")
    def cli(in_directory, out_directory, verbose):
        """Constructs a Graphical User Interface for manually 
        annotating images for the PEATRAIN.PY training dataset.

        PEALABEL.PY generates a Graphical User Interface that 
        successively views images from the IN_DIRECTORY. For each 
        image, the user clicks to place boxes to be cropped and 
        sorted into a training dataset in the OUT_DIRECTORY. The 
        image crops are sorted into two categories:  

            - LEFT CLICKING designates a NEGATIVE to the cropped 
              image (i.e. the Machine Learning algorithm should use 
              the image as a negative example),

            - RIGHT CLICKING designates a POSITIVE to the cropped 
              image (i.e. the Machine Learning algorithm should use 
              the image as a positive example).

        The user can also press BACKSPACE to undo designations. 
        Pressing the SPACEBAR moves to the next image if there is 
        one. Pressing ESC ends the annotating.

        Once the sorting has completed, the cropped images in the 
        OUT_DIRECTORY will be labelled with their designation (0 for
        negative, 1 for positive) and bounding box coordinates. For 
        convenience, this label information is also written to the 
        output file ANNOTATIONS.CSV in a separate column.
        """
        peaLabler(in_directory, out_directory, verbose)

        if (verbose):
            pu.log("cli() function finished. Exiting script.")
            
        sys.exit()
    cli()