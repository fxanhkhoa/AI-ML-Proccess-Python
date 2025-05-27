# In a file like image_storage.py
base64_image = None

def set_image(base64_string):
    global base64_image
    base64_image = base64_string

def get_image():
    return base64_image