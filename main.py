import cv2
import numpy as np
from collections import deque


# Get new Deque
def get_deque(length=1024):
    return deque(maxlen=length)


# Helper function for canvas setup
def trackbar_callback(x):
    print("Trackbar created with value:", x)


# Show the image
def show_frame(window_name, frame):
    cv2.imshow(window_name, frame)


# Create named windows
def create_windows(names, size, positions):
    for idx, name in enumerate(names):
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(name, size[0], size[1])
        cv2.moveWindow(name, positions[idx][0], positions[idx][1])


# Set up the color detection trackbar
def setup_trackbar(window_name):
    # Red
    # cv2.createTrackbar("Max Hue", window_name, 180, 180, trackbar_callback)
    # cv2.createTrackbar("Max Saturation", window_name, 255, 255, trackbar_callback)
    # cv2.createTrackbar("Max Value", window_name, 255, 255, trackbar_callback)
    # cv2.createTrackbar("Min Hue", window_name, 170, 180, trackbar_callback)
    # cv2.createTrackbar("Min Saturation", window_name, 100, 255, trackbar_callback)
    # cv2.createTrackbar("Min Value", window_name, 50, 255, trackbar_callback)

    # Yellow
    # cv2.createTrackbar("Max Hue", window_name, 30, 180, trackbar_callback)
    # cv2.createTrackbar("Max Saturation", window_name, 255, 255, trackbar_callback)
    # cv2.createTrackbar("Max Value", window_name, 255, 255, trackbar_callback)
    # cv2.createTrackbar("Min Hue", window_name, 20, 180, trackbar_callback)
    # cv2.createTrackbar("Min Saturation", window_name, 75, 255, trackbar_callback)
    # cv2.createTrackbar("Min Value", window_name, 75, 255, trackbar_callback)

    # Blue
    cv2.createTrackbar("Max Hue", window_name, 130, 180, trackbar_callback)
    cv2.createTrackbar("Max Saturation", window_name, 255, 255, trackbar_callback)
    cv2.createTrackbar("Max Value", window_name, 255, 255, trackbar_callback)
    cv2.createTrackbar("Min Hue", window_name, 110, 180, trackbar_callback)
    cv2.createTrackbar("Min Saturation", window_name, 80, 255, trackbar_callback)
    cv2.createTrackbar("Min Value", window_name, 80, 255, trackbar_callback)


# Set up the canvas and return it along with kernel
def setup_color_buttons(data, color_list, texts, show_text=False):
    white_color = (255, 255, 255)

    data = cv2.rectangle(data, (25, 0), (135, 60), color_list[0], 1)
    cv2.putText(data, texts[0], (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_list[0], 2, cv2.LINE_AA)

    data = cv2.rectangle(data, (145, 0), (255, 60), color_list[1], -1)
    if show_text:
        cv2.putText(data, texts[1], (160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2, cv2.LINE_AA)

    data = cv2.rectangle(data, (265, 0), (375, 60), color_list[2], -1)
    if show_text:
        cv2.putText(data, texts[2], (280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2, cv2.LINE_AA)

    data = cv2.rectangle(data, (385, 0), (495, 60), color_list[3], -1)
    if show_text:
        cv2.putText(data, texts[3], (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2, cv2.LINE_AA)

    data = cv2.rectangle(data, (505, 0), (615, 60), color_list[4], -1)
    if show_text:
        cv2.putText(data, texts[4], (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2, cv2.LINE_AA)

    return data


# Given the camera, captures the next frame and
# returns original frame, flipped frame and hsv frame
def get_frame(cam, resize_ratio=2):
    ret, frame = cam.read()
    h, w, c = frame.shape
    frame = cv2.resize(frame, (w//resize_ratio, h//resize_ratio), interpolation=cv2.INTER_AREA)
    flipped_frame = cv2.flip(frame, 1)
    hsv_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2HSV)

    return frame, flipped_frame, hsv_frame, ret


# Get the upper and lower HSVs
def get_target_hsv(window_name):
    upper_hsv = np.array([cv2.getTrackbarPos("Max Hue", window_name), cv2.getTrackbarPos("Max Saturation", window_name), cv2.getTrackbarPos("Max Value", window_name)])
    lower_hsv = np.array([cv2.getTrackbarPos("Min Hue", window_name), cv2.getTrackbarPos("Min Saturation", window_name), cv2.getTrackbarPos("Min Value", window_name)])
    return upper_hsv, lower_hsv


# Get the detection mask and contour of the object/pen
def get_mask_and_contour(frame, max_hsv, min_hsv):
    # show_frame("HSV", frame)
    hsv_mask = cv2.inRange(frame, min_hsv, max_hsv)
    # show_frame("Mask", mask)
    hsv_mask = cv2.erode(hsv_mask, kernel, iterations=1)
    # show_frame("Erode", mask)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
    # show_frame("MorphEx", mask)
    hsv_mask = cv2.dilate(hsv_mask, kernel, iterations=1)
    # show_frame("Dilate", mask)

    # Find the contours for the object/pen
    contur, _ = cv2.findContours(hsv_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return hsv_mask, contur


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

# Set the canvas up and windows
create_windows(window_names, window_size, window_positions)
setup_trackbar(color_detector)
canvas = setup_color_buttons(canvas, colors, color_names, show_text=True)

# Points for drawing the line with their count of elements
red_points = [get_deque()]
red_count = 0
green_points = [get_deque()]
green_count = 0
blue_points = [get_deque()]
blue_count = 0
purple_points = [get_deque()]
purple_count = 0

# Start the default camera
camera = cv2.VideoCapture(0)
width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Print the screen resolution
print("Screen resolution: {}x{}".format(int(width), int(height)))

# Continue capturing frames and doing things
while True:
    # Get the required frames
    main_frame, flipped_frame, hsv_frame, ret = get_frame(camera, resize_ratio=2)

    # Get the upper and lower HSV values of target color
    u_hsv, l_hsv = get_target_hsv(color_detector)

    # Put color selection options over the frames
    flipped_frame = setup_color_buttons(flipped_frame, colors, color_names, show_text=True)

    # Detect the object/pen
    mask, contour = get_mask_and_contour(hsv_frame, u_hsv, l_hsv)
    has_contour = len(contour) > 0

    # Draw the contour if it has one
    # And follow as per selection
    if has_contour:
        # Find the biggest contour
        c = sorted(contour, key=cv2.contourArea, reverse=True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Draw the circle showing the contour
        cv2.circle(flipped_frame, (int(x), int(y)), int(radius), (255, 0, 127), 2)
        moments = cv2.moments(c)
        center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))

        # If we are over the color selecting boxes
        if center[1] <= 60:
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
                # Clear the canvas, keep the top selectors
                canvas[61:, :, :] = 255

                # Reset all the values
                red_points = [get_deque()]
                red_count = 0
                green_points = [get_deque()]
                green_count = 0
                blue_points = [get_deque()]
                blue_count = 0
                purple_points = [get_deque()]
                purple_count = 0
        else:
            # Else, draw over the canvas
            if selected_color == 1:
                red_points[red_count].appendleft(center)
            elif selected_color == 2:
                green_points[green_count].appendleft(center)
            elif selected_color == 3:
                blue_points[blue_count].appendleft(center)
            elif selected_color == 4:
                purple_points[purple_count].appendleft(center)

    # If nothing detected, push empty deque
    else:
        red_points.append(get_deque())
        red_count += 1
        green_points.append(get_deque())
        green_count += 1
        blue_points.append(get_deque())
        blue_count += 1
        purple_points.append(get_deque())
        purple_count += 1

    # Draw over the canvas and frame
    points = [red_points, green_points, blue_points, purple_points]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(flipped_frame, points[i][j][k - 1], points[i][j][k], colors[i+1], 2)
                cv2.line(canvas, points[i][j][k - 1], points[i][j][k], colors[i+1], 2)

    # Show all the windows
    cv2.imshow(mask_window, mask)
    cv2.imshow(pen_tracking_window, flipped_frame)
    cv2.imshow(canvas_window, canvas)

    # Stop and break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release all resources
camera.release()
cv2.destroyAllWindows()
