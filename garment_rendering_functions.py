# garment_rendering_functions.py

from PIL import Image

def load_garment(garment_path, target_width, target_height):
    """
    Loads the garment image and resizes it to fit a specific target width and height.
    """
    garment = Image.open(garment_path).convert("RGBA")
    garment = garment.resize((target_width, target_height), Image.ANTIALIAS)
    return garment

def position_garment(user_image, garment_image, x_offset=0, y_offset=0):
    """
    Positions the garment image onto the user image at the specified x and y offsets.
    The garment image is assumed to be the appropriate size for the desired fit.
    """
    # Create a new image for combining
    combined_image = user_image.copy()

    # Calculate position based on offset values
    position = (x_offset, y_offset)

    # Paste the garment image onto the user image with transparency
    combined_image.paste(garment_image, position, garment_image)
    return combined_image

def overlay_garment(user_image_path, garment_image_path, garment_width_ratio=0.5, x_offset=0, y_offset=0):
    """
    Loads the user and garment images, resizes the garment based on the user image size,
    and positions the garment onto the user image.
    
    Parameters:
    - garment_width_ratio: Proportion of the user image width that the garment should cover.
    - x_offset, y_offset: Positional adjustments for garment placement.
    """
    # Load and preprocess the user image
    user_image = Image.open(user_image_path).convert("RGBA")

    # Calculate garment size based on user image dimensions
    garment_width = int(user_image.width * garment_width_ratio)
    garment_height = int(garment_width * 1.2)  # Adjust height as needed for the garment's aspect ratio

    # Load and resize the garment image
    garment_image = load_garment(garment_image_path, garment_width, garment_height)

    # Position and overlay the garment on the user image
    combined_image = position_garment(user_image, garment_image, x_offset, y_offset)
    return combined_image
