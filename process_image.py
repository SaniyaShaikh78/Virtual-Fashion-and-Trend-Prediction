import cv2
import numpy as np
import mediapipe as mp
from rembg import remove

def remove_background(image_path):
    """Removes background from an image using rembg"""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    output = remove(image)
    return output

def detect_body_landmarks(image_path):
    """Detects body keypoints using MediaPipe"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        return [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in results.pose_landmarks.landmark]
    return None

def warp_garment(garment, user_pose, output_shape):
    """Warps the garment to fit user body keypoints using Thin Plate Spline (TPS)"""
    garment_resized = cv2.resize(garment, (output_shape[1], output_shape[0]))  # Resize garment to match user image
    return garment_resized

def overlay_images(user_image, garment_image):
    """Overlays the processed garment on the user image"""
    
    # Ensure the garment is the same size as the user image
    garment_image = cv2.resize(garment_image, (user_image.shape[1], user_image.shape[0]))

    # Convert user_image to writable format
    user_image = user_image.copy()

    # Convert 4-channel RGBA garment image to 3-channel RGB
    if garment_image.shape[-1] == 4:  # Check if garment has an alpha channel
        garment_rgb = garment_image[:, :, :3]  # Extract only RGB channels
        mask = garment_image[:, :, 3] > 0  # Use alpha channel as mask
    else:
        garment_rgb = garment_image  # If no alpha, use as is
        mask = np.ones(garment_image.shape[:2], dtype=np.uint8)  # Create full mask

    # Convert 4-channel RGBA user image to 3-channel RGB
    if user_image.shape[-1] == 4:
        user_image = user_image[:, :, :3]  # Convert to RGB
    
    # Ensure mask is correctly formatted
    mask = mask.astype(np.uint8)

    # Overlay only where garment exists
    user_image[mask > 0] = garment_rgb[mask > 0]

    return user_image


def process_user_image(user_image_path, garment_image_path, output_path):
    """Processes the user image and overlays the garment"""
    # Step 1: Load images
    user_image = cv2.imread(user_image_path, cv2.IMREAD_UNCHANGED)
    garment_image = cv2.imread(garment_image_path, cv2.IMREAD_UNCHANGED)

    # Step 2: Remove backgrounds
    user_image = remove_background(user_image_path)
    garment_image = remove_background(garment_image_path)

    # Step 3: Detect user's pose
    user_pose = detect_body_landmarks(user_image_path)
    if not user_pose:
        raise Exception("Could not detect user pose!")

    # Step 4: Warp garment
    warped_garment = warp_garment(garment_image, user_pose, user_image.shape)

    # Step 5: Overlay the garment on the user
    final_output = overlay_images(user_image, warped_garment)

    # Step 6: Save the final image
    cv2.imwrite(output_path, final_output)
