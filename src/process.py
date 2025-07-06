from asyncio import sleep
import base64
import cv2
import numpy as np
from socketio import AsyncServer
from src.image_storage import get_image
from src.type.engine import EngineUpdate
import math
import time
from src.util.image import is_circular
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras import layers

# Global variables to track window state
window_created = False
window_track_bar_created = False
window_sign_created = False

engine = EngineUpdate(acceleration=0.0, steerAngle=0, brake=0.0)
print_output = False

accel_state = 1  # 1 for accelerating, 0 for not
accel_last_switch = time.time()

threshold1 = 95  # Initial value for threshold1
threshold2 = 50  # Initial value for threshold2
threshold3 = 150  # Example third threshold
lane_line_threshold = 150

IMAGE_SIZE = (32, 32)
input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
num_classes = 58  # Number of traffic sign classes, adjust as needed
model = None

entryFound = False

def define_keras_model():
    """Define a simple Keras model for traffic sign detection."""
    model = keras.Sequential([
         # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25), # Added Dropout after the first pooling layer

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25), # Added Dropout after the second pooling layer

        # Flattening and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Added Dropout after the first Dense layer
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

def initialize():
    global model
    model = define_keras_model()
    model.load_weights('./traffic_sign.weights.h5')


initialize()

async def process_start(sio: AsyncServer, client_id: str):
    global window_created, last_image, accel_state, accel_last_switch
    global threshold1, threshold2, threshold3, window_track_bar_created
    global lane_line_threshold, entryFound

    # --- UI with sliders ---
    if not window_track_bar_created:
        cv2.namedWindow('track bar', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Threshold 1', 'track bar', threshold1, 255, lambda v: None)
        cv2.createTrackbar('Threshold 2', 'track bar', threshold2, 255, lambda v: None)
        cv2.createTrackbar('Threshold 3', 'track bar', threshold3, 255, lambda v: None)
        cv2.createTrackbar('Aperture', 'track bar', 7, 7, lambda v: None)
        cv2.createTrackbar('lane_line_threshold', 'track bar', lane_line_threshold, 255, lambda v: None)
        window_track_bar_created = True

    # Get current slider values
    threshold1 = cv2.getTrackbarPos('Threshold 1', 'track bar')
    threshold2 = cv2.getTrackbarPos('Threshold 2', 'track bar')
    threshold3 = cv2.getTrackbarPos('Threshold 3', 'track bar')
    aperture = cv2.getTrackbarPos('Aperture', 'track bar')
    lane_line_threshold = cv2.getTrackbarPos('lane_line_threshold', 'track bar')
    if aperture % 2 == 0:  # Ensure aperture is odd and >=3
        aperture = max(3, aperture + 1)


    # Get the latest base64 image
    base64_string = get_image()
    if not base64_string:
        # If no image is available, keep showing the last frame if it exists
        if window_created and last_image is not None:
            cv2.imshow('Video Stream', last_image)
            cv2.waitKey(1)  # Short wait to update the UI
        return
    # Process the base64 string

    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]

    try:
        # Decode the base64 string
        img_data = base64.b64decode(base64_string)

        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)

        # Decode the image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return
            
        # Store the current image
        last_image = image.copy()
        
        # Create window if it doesn't exist
        if not window_created:
            cv2.namedWindow('Video Stream', cv2.WINDOW_NORMAL)
            window_created = True


        # ------------------------------- MAIN PROCESSING -------------------------------------
        # Calculate the height of the image
        height, width = image.shape[:2]
        # Crop to the bottom third
        cropped_image = image[2 * height // 3:height, 0:width]
        # Convert to grayscale
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blur_gray = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # Apply thresholding (simple binary thresholding)
        # Adjust the threshold value (e.g., 100) and max_value (e.g., 255) as needed
        ret, thresh_image = cv2.threshold(blur_gray, lane_line_threshold, 255, cv2.THRESH_BINARY)
        thresh_image = cv2.bitwise_not(thresh_image)

        contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the two largest contours (assuming these are the lane lines)
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Log and draw all contours
        if print_output:
            print(f"Found {len(contours)} contours.")
        # Draw all contours for visualization (optional)
        all_contours_image = cropped_image.copy()
        cv2.drawContours(all_contours_image, contours, -1, (0, 255, 0), 2) # Draw all contours in green

        # The lane line is the max contour
        max_contour = contours[0]
        if print_output:
            print(f"Largest contour has area: {cv2.contourArea(max_contour)}")

        if len(contours) > 100:
            lane_line_threshold = 140
            cv2.setTrackbarPos('lane_line_threshold', 'track bar', lane_line_threshold)

        # You can then draw just this largest contour if needed
        max_contour_image = cropped_image.copy()
        cv2.drawContours(max_contour_image, [max_contour], -1, (255, 0, 0), 2) # Draw in blue

        M = cv2.moments(max_contour)
        # Ensure M['m00'] is not zero to avoid division by zero
        if M['m00'] != 0:
            # Calculate the centroid (center of mass)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            # Draw a circle at the center
            cv2.circle(max_contour_image, (cX, cY), 5, (0, 255, 255), -1) # Draw center
            # Define the text to display
            center_text = f"({cX}, {cY}), {len(contours)} contours"

            # Draw the text next to the center
            # The last argument is the text color (BGR: Blue, Green, Red)
            # The font scale and thickness can be adjusted for better visibility
            cv2.putText(max_contour_image, center_text, (cX + 10, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
        # The bottom center point of the image
        bottom_center_x = width // 2
        bottom_center_y = cropped_image.shape[0] # The bottom edge is at the height of the cropped image

        # Calculate the difference in x and y coordinates
        delta_x = cX - bottom_center_x
        delta_y = cY - bottom_center_y # Note: Y increases downwards in image coordinates

        # Calculate the angle in radians using arctan2
        # arctan2 handles all four quadrants correctly and avoids division by zero issues
        # It calculates the angle between the positive x-axis and the point (x, y)
        # In our case, the vector is from the bottom center (bottom_center_x, bottom_center_y)
        # to the centroid (cX, cY).
        # The angle is measured from the horizontal line passing through the bottom center.
        # To get the angle from the vertical line (which represents straight ahead),
        # we can use arctan2(x, y) which calculates the angle from the positive y-axis.
        # The positive y-axis points downwards in image coordinates.
        # We want the angle from the upward vertical, so we use -delta_y.
        deviation_angle_rad = math.atan2(delta_x, -delta_y)

        # Convert the angle from radians to degrees
        deviation_angle_deg = math.degrees(deviation_angle_rad)

        if print_output:
            print(f"Centroid of the largest contour: ({cX}, {cY})")
            print(f"Bottom center of the image: ({bottom_center_x}, {bottom_center_y})")
            print(f"Deviation angle (radians): {deviation_angle_rad}")
            print(f"Deviation angle (degrees): {deviation_angle_deg}")
            print(f"need to set angle (degrees): {deviation_angle_deg / 2}")

        # Optional: Draw a line from the bottom center to the centroid on the max contour image
        # Make sure max_contour_image is still available or recreate it if needed
        line_image = max_contour_image.copy()
        cv2.line(line_image, (bottom_center_x, bottom_center_y), (cX, cY), (255, 0, 255), 2) # Draw in magenta

        # Add text for the deviation angle
        angle_text = f"Angle: {deviation_angle_deg:.2f} deg"
        # Position the text near the bottom center or the centroid
        text_pos = (bottom_center_x + 20, bottom_center_y - 20) # Adjust position as needed
        cv2.putText(line_image, angle_text, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        

        
        # --- Process Traffic Sign Detection ---
        traffic_image = last_image.copy()
        traffic_gray = cv2.cvtColor(traffic_image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        traffic_gray_blur = cv2.GaussianBlur(traffic_gray, (5, 5), 0)
        # Apply thresholding (simple binary thresholding)
        # Adjust the threshold value (e.g., 100) and max_value (e.g., 255) as needed
        ret, thresh_image_traffic = cv2.threshold(traffic_gray_blur, threshold1, 255, cv2.THRESH_BINARY)

        # Canny edge detection on the thresholded image
        edges = cv2.Canny(thresh_image_traffic, threshold2, threshold3, apertureSize=aperture, L2gradient=True)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Iterate through the contours and check for squares
        cicular_contours = []
        for contour in contours:
            if is_circular(contour):
                cicular_contours.append(contour)

        if cicular_contours:
            # Find the largest circular contour by area
            biggest_circular = max(cicular_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(biggest_circular)
            # Add padding (e.g., 10 pixels)
            padding = 10
            x_pad = max(x - padding, 0)
            y_pad = max(y - padding, 0)
            x2_pad = min(x + w + padding, traffic_image.shape[1])
            y2_pad = min(y + h + padding, traffic_image.shape[0])
            # Crop the region of interest from the original (color) traffic image with padding
            keras_roi = traffic_image[y_pad:y2_pad, x_pad:x2_pad].copy()
            # Optional: resize for keras model input, e.g. (32, 32)
            keras_input = cv2.resize(keras_roi, IMAGE_SIZE)
            # keras_input is now ready for keras model prediction

            img_array = keras.utils.img_to_array(keras_input)
            img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis
            predictions = model.predict(img_array)
            found_index = np.argmax(predictions[0])
            print(f"Predicted index: {found_index}, Confidence: {predictions[0][found_index]}")
            sign_text = ""
            if found_index == 29 and not entryFound:
                print("Traffic sign detected: No Entry")
                sign_text = "No Entry"
                engine.acceleration = 0
                engine.steerAngle = 0
                engine.brake = 1

                await engine.emit_to_simulation(sio, client_id)
                await sleep(5)  # Sleep to simulate processing time
                entryFound = True
            elif found_index == 12:
                print("Traffic sign detected: Don't turn left or right")
                sign_text = "Don't turn left or right"
                threshold1 = 150
                lane_line_threshold = 115
                cv2.setTrackbarPos('Threshold 1', 'track bar', threshold1)
                cv2.setTrackbarPos('lane_line_threshold', 'track bar', lane_line_threshold)

            if sign_text:
                text_pos = (x - 100, max(y - 10, 0))  # 10 pixels above the box, not less than 0
                cv2.putText(
                    traffic_image, sign_text, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
                )

            # For debug: show the cropped region
            cv2.imshow('Biggest Circular ROI', keras_input)
        else:
            keras_input = None  # No circular contour found

        # --- END Process Traffic Sign Detection ---

        # # Draw the detected squares on the original image
        cv2.drawContours(traffic_image, cicular_contours, -1, (0, 255, 0), 2)

        # --- Acceleration interval logic 3s acceleration = 1, 2s acceleration = 0 ---
        now = time.time()
        if accel_state == 1 and now - accel_last_switch >= 2:
            accel_state = 0
            accel_last_switch = now
        elif accel_state == 0 and now - accel_last_switch >= 2:
            accel_state = 1
            accel_last_switch = now

        engine.acceleration = accel_state
        engine.steerAngle = deviation_angle_deg / 2
        engine.brake = 0

        await engine.emit_to_simulation(sio, client_id)
        
        # ------------------------------- END MAIN PROCESSING -------------------------------------

        # Display the image
        cv2.imshow('Video Stream', line_image)
        cv2.imshow('Traffic_sign', traffic_image)
        
        # Use a short wait time to make it feel like video
        # 1ms is enough to update the UI without blocking too long
        key = cv2.waitKey(1)
        
        # Add a way to exit the window with 'q' key
        if key == ord('q'):
            cv2.destroyAllWindows()
            window_created = False
            
    except Exception as e: 
        print(f"Error processing image: {e}")


def cleanup():
    """Call this when shutting down the application"""
    global window_created
    if window_created:
        cv2.destroyAllWindows()
        window_created = False