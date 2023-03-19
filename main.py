import cv2
import numpy as np
from collections import deque


# Get new Deque
def get_deque(length=1024):
    return deque(maxlen=length)


# Helper function for canvas setup
def trackbar_callback(value):
    print("Trackbar created with value:", value)


# Show the given frame
def show_frame(window_name, frame):
    cv2.imshow(window_name, frame)


# Create named windows for showing the canvas, mask and original frame
def create_windows(names, size, positions):
    for idx, name in enumerate(names):
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(name, size[0], size[1])
        cv2.moveWindow(name, positions[idx][0], positions[idx][1])


# Set up the color detection trackbar
# Predefined colors are red, yellow and blue
# Reference: https://docs.opencv.org/3.4/da/d6a/tutorial_trackbar.html
def setup_trackbar(window_name, target_color='blue'):
    if target_color == 'red':
        # Set min target values
        cv2.createTrackbar("Min Hue", window_name, 170, 180, trackbar_callback)
        cv2.createTrackbar("Min Saturation", window_name, 100, 255, trackbar_callback)
        cv2.createTrackbar("Min Value", window_name, 50, 255, trackbar_callback)

        # Set max target values
        cv2.createTrackbar("Max Hue", window_name, 180, 180, trackbar_callback)
        cv2.createTrackbar("Max Saturation", window_name, 255, 255, trackbar_callback)
        cv2.createTrackbar("Max Value", window_name, 255, 255, trackbar_callback)

    elif target_color == 'yellow':
        # Set min target values
        cv2.createTrackbar("Min Hue", window_name, 20, 180, trackbar_callback)
        cv2.createTrackbar("Min Saturation", window_name, 75, 255, trackbar_callback)
        cv2.createTrackbar("Min Value", window_name, 75, 255, trackbar_callback)

        # Set max target values
        cv2.createTrackbar("Max Hue", window_name, 30, 180, trackbar_callback)
        cv2.createTrackbar("Max Saturation", window_name, 255, 255, trackbar_callback)
        cv2.createTrackbar("Max Value", window_name, 255, 255, trackbar_callback)

    elif target_color == 'blue':
        # Set min target values
        cv2.createTrackbar("Min Hue", window_name, 110, 180, trackbar_callback)
        cv2.createTrackbar("Min Saturation", window_name, 80, 255, trackbar_callback)
        cv2.createTrackbar("Min Value", window_name, 80, 255, trackbar_callback)

        # Set max target values
        cv2.createTrackbar("Max Hue", window_name, 130, 180, trackbar_callback)
        cv2.createTrackbar("Max Saturation", window_name, 255, 255, trackbar_callback)
        cv2.createTrackbar("Max Value", window_name, 255, 255, trackbar_callback)


# Given an image frame, it sets up the action menu buttons
# And returns the updated frame with the menu items on it
# Reference: https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
def setup_color_buttons(data, color_list, texts, show_text=False):
    white_color = (255, 255, 255)

    # Set up the first element
    data = cv2.rectangle(data, (25, 0), (135, 60), color_list[0], 1)
    cv2.putText(data, texts[0], (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_list[0], 2, cv2.LINE_AA)

    # Set up the second element
    data = cv2.rectangle(data, (145, 0), (255, 60), color_list[1], -1)
    if show_text:
        cv2.putText(data, texts[1], (160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2, cv2.LINE_AA)

    # Set up the third element
    data = cv2.rectangle(data, (265, 0), (375, 60), color_list[2], -1)
    if show_text:
        cv2.putText(data, texts[2], (280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2, cv2.LINE_AA)

    # Set up the forth element
    data = cv2.rectangle(data, (385, 0), (495, 60), color_list[3], -1)
    if show_text:
        cv2.putText(data, texts[3], (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2, cv2.LINE_AA)

    # Set up the fifth element
    data = cv2.rectangle(data, (505, 0), (615, 60), color_list[4], -1)
    if show_text:
        cv2.putText(data, texts[4], (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2, cv2.LINE_AA)

    return data


# Given the camera, captures the next frame and
# returns original frame, flipped frame and hsv frame
def get_frame(cam, resize_ratio=2):
    # Capture the frame from camera
    _, frame = cam.read()
    h, w, channel = frame.shape

    # Resize the frame
    frame = cv2.resize(frame, (w//resize_ratio, h//resize_ratio), interpolation=cv2.INTER_AREA)

    # Flip the frame and convert to HSV
    frame_flipped = cv2.flip(frame, 1)
    frame_hsv = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2HSV)

    # Return data
    return frame, frame_flipped, frame_hsv


# Get the upper and lower HSVs
def get_target_hsv(window_name):
    upper_hsv = np.array([cv2.getTrackbarPos("Max Hue", window_name), cv2.getTrackbarPos("Max Saturation", window_name), cv2.getTrackbarPos("Max Value", window_name)])
    lower_hsv = np.array([cv2.getTrackbarPos("Min Hue", window_name), cv2.getTrackbarPos("Min Saturation", window_name), cv2.getTrackbarPos("Min Value", window_name)])
    return upper_hsv, lower_hsv


# Get the detection mask and contour of the object/pen
# Reference: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
def get_mask_and_contour(frame, max_hsv, min_hsv):
    # Creating Mask in HSV color space
    # show_frame("HSV", frame)
    hsv_mask = cv2.inRange(frame, min_hsv, max_hsv)
    # show_frame("Mask", mask)

    # Apply an Erode operation
    hsv_mask = cv2.erode(hsv_mask, kernel, iterations=1)
    # show_frame("Erode", mask)

    # Apply a Morphological Open operation
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
    # show_frame("MorphEx", mask)

    # Apply a Dilate operation
    hsv_mask = cv2.dilate(hsv_mask, kernel, iterations=1)
    # show_frame("Dilate", mask)

    # Find the contours for the object/pen
    contur, _ = cv2.findContours(hsv_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return hsv_mask, contur


# Draw on the canvas based on the marker position
# And selected color
def draw_over_canvas():
    for idx in range(len(rgbp_points)):
        for j in range(len(rgbp_points[idx])):
            for k in range(1, len(rgbp_points[idx][j])):
                if rgbp_points[idx][j][k - 1] is not None and rgbp_points[idx][j][k] is not None:
                    cv2.line(flipped_frame, rgbp_points[idx][j][k - 1], rgbp_points[idx][j][k], colors[idx+1], 2)
                    cv2.line(canvas, rgbp_points[idx][j][k - 1], rgbp_points[idx][j][k], colors[idx+1], 2)


# Clear the canvas and reset the color points
# Return the updated values
def reset_canvas(active_canvas):
    # Clear the canvas, keep the top selectors
    active_canvas[61:, :, :] = 255

    # Reset all the values
    color_points = [
        [get_deque()],          # For red
        [get_deque()],          # For green
        [get_deque()],          # For blue
        [get_deque()],          # For purple
    ]
    color_indices = [0, 0, 0, 0]
    selected_color_idx = 1

    # Return the updated values
    return canvas, color_points, color_indices, selected_color_idx


# Required parameters
pen_tracking_window = "Pen Tracker"
mask_window = "Mask"
canvas_window = "Canvas"
color_detector = "Target Color"
window_names = [color_detector, canvas_window, pen_tracking_window, mask_window]
window_positions = [(40, 400), (0, 0), (660, 0), (660, 380)]
colors = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (128, 0, 128)]
color_names = ["Clear", "Red", "Green", "Blue", "Purple"]
window_size = (640, 360)
selected_color = 1
canvas = np.ones((window_size[1], window_size[0], 3), np.uint8) * 255
kernel = np.ones((5, 5), np.uint8)

# Points for drawing the line with their count of elements
rgbp_points = [
    [get_deque()],          # For red
    [get_deque()],          # For green
    [get_deque()],          # For blue
    [get_deque()],          # For purple
]
rgbp_counts = [0, 0, 0, 0]

# Start the default camera
# Reference: https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Cannot open camera")
    exit()

# Print the screen resolution
width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Screen resolution: {}x{}".format(int(width), int(height)))

# Set the canvas up and windows
create_windows(window_names, window_size, window_positions)
canvas = setup_color_buttons(canvas, colors, color_names, show_text=True)
setup_trackbar(color_detector)

# Continue capturing frames and doing things
while True:
    # Get the required frames
    main_frame, flipped_frame, hsv_frame = get_frame(camera, resize_ratio=2)

    # Put color selection options over the frames
    flipped_frame = setup_color_buttons(flipped_frame, colors, color_names, show_text=True)

    # Get the upper and lower HSV values of target color
    u_hsv, l_hsv = get_target_hsv(color_detector)

    # Detect the object/pen
    mask, contour = get_mask_and_contour(hsv_frame, u_hsv, l_hsv)
    has_contour = len(contour) > 0

    # Draw the contour if it has one
    # And follow as per selection
    if has_contour:
        # Find the biggest contour
        contour_points_sorted = sorted(contour, key=cv2.contourArea, reverse=True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(contour_points_sorted)

        # Find the center of the contour
        moments = cv2.moments(contour_points_sorted)
        center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))

        # Draw the circle showing the contour
        cv2.circle(flipped_frame, (int(x), int(y)), int(radius), (255, 0, 127), 2)

        # If we are over the color selecting boxes
        if center[1] > 60:
            # Draw over the canvas
            rgbp_points[selected_color-1][rgbp_counts[selected_color-1]].appendleft(center)
        else:
            # Menu Selected, perform operation accordingly
            if 145 <= center[0] <= 255:
                # Red selected
                selected_color = 1
            elif 265 <= center[0] <= 375:
                # Green selected
                selected_color = 2
            elif 385 <= center[0] <= 495:
                # Blue selected
                selected_color = 3
            elif 505 <= center[0] <= 615:
                # Purple selected
                selected_color = 4
            # We are over the clear menu
            elif 25 <= center[0] <= 135:
                # Reset the canvas and color points
                canvas, rgbp_points, rgbp_counts, selected_color = reset_canvas(canvas)

    # If nothing detected
    else:
        # Push empty deque and increment count to maintain consistency
        for i in range(4):
            rgbp_points[i].append(get_deque())
            rgbp_counts[i] += 1

    # Draw over the canvas and frame
    draw_over_canvas()

    # Show all the windows
    cv2.imshow(canvas_window, canvas)
    cv2.imshow(mask_window, mask)
    cv2.imshow(pen_tracking_window, flipped_frame)

    # Stop and break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release all resources
camera.release()
cv2.destroyAllWindows()
