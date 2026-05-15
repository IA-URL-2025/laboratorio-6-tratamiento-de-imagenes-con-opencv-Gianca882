import cv2
import numpy as np


def to_grayscale(image):
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image, width, height):

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def apply_blur(image, kernel_size=5):

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def adjust_brightness_contrast(image, alpha=1.0, beta=0):

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)



def apply_threshold(image, thresh_value=127):

    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, result = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    return result


def detect_edges(image, low=50, high=150):

    if len(image.shape)>2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(image, low, high)


def full_pipeline(image, target_width=224, target_height=224):

    result = resize_image(image, target_width, target_height)
    result = to_grayscale(result)
    result = apply_blur(result, kernel_size=3)
    result = detect_edges(result, low=50, high=150)
    return result
