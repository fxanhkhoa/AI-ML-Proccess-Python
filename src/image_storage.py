# In a file like image_storage.py
base64_image = None
traffic_sign_image = None

def set_image(base64_string):
    global base64_image
    base64_image = base64_string

def get_image():
    return base64_image

def set_traffic_sign_image(base64_string):
    global traffic_sign_image
    traffic_sign_image = base64_string

def get_traffic_sign_image():
    return traffic_sign_image