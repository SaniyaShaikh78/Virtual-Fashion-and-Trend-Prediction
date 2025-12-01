# preprocessor.py

from PIL import Image, ImageOps
import cv2
import numpy as np

def resize_image(image_path, target_size=(400, 600)):
    """
    Resizes the input image to a specified target size while maintaining aspect ratio.
    Fills any empty space with a solid background color to match the target size.
    """
    image = Image.open(image_path).convert("RGBA")
    image.thumbnail(target_size, Image.ANTIALIAS)  # Resize while maintaining aspect ratio

    # Create a background image and paste the resized image onto it
    background = Image.new("RGBA", target_size, (255, 255, 255, 0))  # Transparent background
    offset = ((target_size[0] - image.width) // 2, (target_size[1] - image.height) // 2)
    background.paste(image, offset)

    return background

def remove_background(image_path):
    """
    Removes the background of the image by applying a mask.
    Note: This requires OpenCV's grabCut method, suitable for portrait-style photos.
    """
    image = cv2.imread(image_path)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Define a rectangle around the object (this can be adjusted for better results)
    rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)

    # Apply grabCut algorithm
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    result = image * mask2[:, :, np.newaxis]

    # Convert to RGBA with transparency
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGBA)
    result[result[:, :, 3] == 0] = [255, 255, 255, 0]  # Transparent background

    return Image.fromarray(result)

def apply_preprocessing(image_path, target_size=(400, 600), remove_bg=False):
    """
    Applies resizing and optional background removal to prepare the image for overlay.
    """
    # Step 1: Resize the image
    processed_image = resize_image(image_path, target_size)

    # Step 2: Optionally remove the background
    if remove_bg:
        processed_image = remove_background(image_path)

    return processed_image

