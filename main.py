import os
import cv2
import numpy as np
from PIL import Image

def remove_white_background(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def center_overlay(background, overlay, y_offset_ratio=0.35):
    bh, bw = background.shape[:2]

    # Resize overlay to cover about 60% width of the background
    target_width = int(bw * 0.6)
    scale_ratio = target_width / overlay.shape[1]
    new_size = (target_width, int(overlay.shape[0] * scale_ratio))
    overlay = cv2.resize(overlay, new_size)
    oh, ow = overlay.shape[:2]

    x = (bw - ow) // 2
    y = int(bh * y_offset_ratio)  # Position near chest

    if y + oh > bh:
        y = bh - oh

    roi = background[y:y+oh, x:x+ow]

    gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(overlay, overlay, mask=mask)
    combined = cv2.add(bg, fg)
    background[y:y+oh, x:x+ow] = combined
    return background

def process_tryon(user_image_path, garment_image_path, output_image_path="static/finalimg.png", background=True):
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    user_img = cv2.imread(user_image_path)
    if user_img is None:
        raise ValueError("User image could not be loaded.")

    user_img = cv2.resize(user_img, (384, 512))

    if garment_image_path and os.path.exists(garment_image_path):
        garment = cv2.imread(garment_image_path)
        if garment is None:
            raise ValueError("Garment image could not be loaded.")

        clean_garment = remove_white_background(garment)
        final = center_overlay(user_img, clean_garment)
    else:
        final = user_img

    cv2.imwrite(output_image_path, final)
    return output_image_path
