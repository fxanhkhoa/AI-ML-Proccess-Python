import base64
import cv2
import numpy as np
from src.image_storage import get_image

# Global variables to track window state
window_created = False
last_image = None

def process_start():
    global window_created, last_image
    
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
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return
            
        # Store the current image
        last_image = img.copy()
        
        # Create window if it doesn't exist
        if not window_created:
            cv2.namedWindow('Video Stream', cv2.WINDOW_NORMAL)
            window_created = True
        
        # Display the image
        cv2.imshow('Video Stream', img)
        
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