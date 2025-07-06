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

num_classes = 58

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


initialize()

async def process_start(sio: AsyncServer, client_id: str):
    global window_created, last_image, accel_state, accel_last_switch

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
        ret, thresh_image = cv2.threshold(blur_gray, 150, 255, cv2.THRESH_BINARY)
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