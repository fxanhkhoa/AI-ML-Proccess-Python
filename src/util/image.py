import cv2
import numpy as np

def is_circular(contour, tolerance=0.2):
    """
    Checks if a contour is approximately a circle.

    Args:
        contour: A contour point array (e.g., from cv2.findContours).
        tolerance: A small float to account for imperfections.

    Returns:
        True if the contour is approximately a circle, False otherwise.
    """
    # Calculate the area and perimeter of the contour
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if area < 200:
        return False

    # If the perimeter is zero, it's not a valid shape
    if perimeter == 0:
        return False

    # Calculate the circularity using the formula: 4 * pi * area / (perimeter * perimeter)
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    # print(area, perimeter, circularity)

    # A perfect circle has a circularity of 1. Check if it's within the tolerance.
    return 1 - tolerance <= circularity <= 1 + tolerance