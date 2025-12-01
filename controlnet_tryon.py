import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector
from huggingface_hub import login

# Login only if needed (token must have "read" permission)
# login("hf_your_actual_token_here")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ControlNet model (OpenPose)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose",
    torch_dtype=torch.float16,
).to(device)

# Load Stable Diffusion Inpainting model with ControlNet
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
).to(device)

# Optional performance enhancements
#pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# OpenPose Detector (for generating pose map)
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

# Main function
def run_tryon(user_image_path, cloth_image_path):
    import PIL.Image
    import numpy as np

    # Load input images
    user_image = PIL.Image.open(user_image_path).convert("RGB").resize((512, 512))
    cloth_image = PIL.Image.open(cloth_image_path).convert("RGB").resize((512, 512))

    # Generate pose (control image)
    control_image = openpose(user_image)

    # Run pipeline
    result = pipe(
        image=user_image,
        control_image=control_image,
        mask_image=None,  # Set to a mask image if doing selective inpainting
        prompt="a person wearing stylish fashionable clothes",
        negative_prompt="blurry, bad quality, distorted",
        num_inference_steps=30,
        guidance_scale=8.0,
    )

    # Save result
    output_image = result.images[0]
    output_path = "static/tryon_result.jpg"
    output_image.save(output_path)

    return output_path, 95.5  # Example accuracy value

