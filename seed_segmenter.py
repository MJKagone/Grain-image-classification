"""
This module implements the segmentation of individual seed images
from video frames.
"""

import numpy as np
import skimage


def separate_seeds(image, *, crop_left_right=(50, 500), threshold=0.295, close_kernel=None, open_kernel=None, label_dilation=5):
    """Returns the individual seed images separated from an imagge

    * image - An image as an RGB Numpy 3D array
    * crop_left_right - The left and right X coordinates to crop at
    * threshold - The threshold value to use (0-1)
    * close_kernel - The kernel to use for morphological closing
    * open_kernel - The kernel to use for morphological opening
    * label_dilation - A radius for dilating the individual seed labels

    Returns: List of 2D image arrays
    """

    # Crop the image
    cropped_image = image[:, crop_left_right[0]:crop_left_right[1]]

    # Threshold the image
    grayscale_image = skimage.color.rgb2gray(cropped_image)
    thresholded_image = grayscale_image > threshold

    # Perform morphological closing followed by opening
    close_kernel = close_kernel or skimage.morphology.disk(3)
    open_kernel = open_kernel or skimage.morphology.disk(2)

    closed = skimage.morphology.closing(thresholded_image, close_kernel)
    closed_opened = skimage.morphology.opening(closed, open_kernel)
    area_opened = skimage.morphology.area_opening(closed_opened, 100)

    # Label the image
    labels = skimage.measure.label(area_opened, 0)
    expanded_labels = skimage.segmentation.expand_labels(labels, label_dilation)

    seed_images = []

    for label in skimage.measure.regionprops(expanded_labels):
        top, left, bottom, right = label.bbox
        # Discard regions which coincide with image bounds
        if top == 0 or left == 0 or bottom == cropped_image.shape[0] or right == cropped_image.shape[1]:
            continue

        mask = label.image_filled
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1).repeat(3, 2)
        cropped = cropped_image[top:bottom, left:right]
        seed_images.append(cropped * mask)

    return seed_images
