"""
This module implements the segmentation of individual seed images
from video frames.
"""

import numpy as np
import skimage


def grayscale_saturated_yellow(image):
    """Grayscales an image based on how saturated, bright and yellow it is

    The formula is saturation * value * (is hue between 0 and 120 degrees?)
    simplified.
    """
    is_yellow = (image[:,:,0] > image[:,:,2]) * (image[:,:,1] > image[:,:,2])
    return (np.max(image, 2) - np.min(image, 2)) * is_yellow


def white_balance(image, divider):
    """Divides the image by a 'white' color, attempting to cancel out color imbalance"""
    image[:,:,0] /= divider[0]
    image[:,:,1] /= divider[1]
    image[:,:,2] /= divider[2]
    return image


def separate_seeds(image, *, crop_left_right=(50, 550), threshold=0.295, close_kernel=None, open_kernel=None, label_dilation=5, median_filter=False, white=None, minimum_area=400):
    """Returns the individual seed images separated from an imagge

    * image - An image as an RGB Numpy 3D array
    * crop_left_right - The left and right X coordinates to crop at
    * threshold - The threshold value to use (0-1)
    * close_kernel - The kernel to use for morphological closing
    * open_kernel - The kernel to use for morphological opening
    * label_dilation - A radius for dilating the individual seed labels
    * median_filter - Whether to apply 3x3 median filtering.
    * white - An approximation of the color white in the image. Used to cancel
              out ambient yellow color.

    Returns: List of 2D image arrays
    """

    # Crop the image
    cropped_original = image[:, crop_left_right[0]:crop_left_right[1]]
    cropped_image = skimage.img_as_float(cropped_original)

    if white is not None:
        cropped_image = white_balance(cropped_image, white)
    if median_filter:
        cropped_image = skimage.filters.median(cropped_image)

    # Threshold the image
    #grayscale_image = skimage.color.rgb2gray(cropped_image)
    grayscale_image = grayscale_saturated_yellow(cropped_image)
    thresholded_image = grayscale_image > threshold

    # Perform morphological closing followed by opening
    close_kernel = close_kernel if close_kernel is not None else skimage.morphology.disk(2)
    open_kernel = open_kernel if open_kernel is not None else skimage.morphology.disk(2)

    closed = skimage.morphology.closing(thresholded_image, close_kernel)
    closed_opened = skimage.morphology.opening(closed, open_kernel)

    # Label the image
    labels = skimage.measure.label(closed_opened)
    expanded_labels = skimage.segmentation.expand_labels(labels, label_dilation)

    seed_images = []

    for label in skimage.measure.regionprops(expanded_labels):
        top, left, bottom, right = label.bbox
        # Discard regions which coincide with image bounds
        if top == 0 or left == 0 or bottom == cropped_image.shape[0] or right == cropped_image.shape[1]:
            continue
        # Discard regions that are too small
        if label.area < minimum_area:
            continue

        mask = label.image_filled
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1).repeat(3, 2)
        cropped = cropped_original[top:bottom, left:right]
        seed_images.append(cropped * mask)

    return seed_images
