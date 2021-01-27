import numpy as np
from scipy.ndimage import binary_fill_holes

def threshold(image_denoised, threshold, fill_holes=False):
    image_thresholded = image_denoised >= threshold 
    
    if fill_holes:
        image_thresholded = binary_fill_holes(image_thresholded)

    return image_thresholded
