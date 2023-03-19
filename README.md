# Airbrush: The Digital Finger Painting Experience

## Environment
>> Python == 3.9.6  
>> numpy == 1.24.2  
>> opencv-contrib-python == 4.7.0.72       


## How to run the project: 
- Make sure your computer has a camera or connected to an external webcam
- Install Python 3.9 or above on your computer
- Open terminal and change the working directory to project directory
- Install the required libraries by running `pip install -r requirements.txt`
- Run the code by running `python main.py`


## Things to note
- The default target marker color is `Blue`. You can change the color by changing the target color range in the `Target Color` window.
- Move the marker slowly or with decent speed. Do not move the marker too fast.
- Press `q` to exit the program.


## Problem statement: 
In this project, we will build a virtual canvas where anyone can draw using
a colored marker on hand. A virtual canvas with a white background will be created with the
option to select colors and clear the canvas. We will use the camera to capture a live video
stream, and then use the movement of our hand as a colored marker to draw on a
virtual canvas. When we move the marker onto the predefined colors, the stroke color of the
marker will be changed. If we put the marker on the "Clear All" option, the canvas will be
cleared. Each frame on the stream will be converted into HSV color space and a mask will be
generated. We will use morphological operations to preprocess the mask for detecting the
marker correctly. The movement of the marker will be detected with a contour and the center
point of the contour will be used to draw/put color on the canvas.

## Proposed work: 
Here is a more detailed step by step description of our proposed task:
- **Environment:** We will need Python, OpenCV, and Numpy. These libraries will help us
capture the live stream from the camera, convert the frames into the HSV color space,
and perform various image processing operations to accomplish our goal.
- **Create a virtual canvas:** Using OpenCV, we will create a white background as the virtual
canvas. The size of the canvas will be based on the captured video stream.
- **Live stream and HSV color space:** We will use OpenCV's VideoCapture class to start the
camera feed and capture the live stream. We will convert each frame from the camera
stream into the HSV color space. The HSV color space is more suitable for color-based
image processing tasks, as it separates the hue, saturation, and value components of the
image. This will help us remove noise, detect the marker movement easily.
- **Generate a mask:** We will use some predefined HSV values to generate a mask. The
mask will be used to detect the color of the marker. We will apply morphological
operations, including dilation and erosion, to the mask to remove noise and improve the
accuracy of the marker detection. This step will also help to remove small and unwanted
regions from the mask.
- **Detecting the marker movement and drawing:** We will find the contour of the marker in
each frame to detect the movement of the marker. We will find the contours of the
regions in the mask that correspond to the marker, and use the center point of the
contour to draw on the virtual canvas.
- **Change color and clear the canvas:** An option to clear the canvas and change color by
detecting the marker placement over the "Clear All" and/or predefined color options.
We can do this by checking the position of the marker and if it's over the "Clear All"
option, we can set the virtual canvas to a white background or check if it is over any
predefined color and change the stroke color accordingly.
- **Repeat the steps:** You will repeat these steps continuously to draw on the virtual canvas
in real-time as the marker moves. You will also need to check for the termination of the
camera feed and handle it properly.