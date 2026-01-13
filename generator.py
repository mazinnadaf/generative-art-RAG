import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
from PIL import Image
import retrieve

def generate_with_style(prompt, style_image_paths):
    """
    Generate an image using SDXL with IP-Adapter style images.
    
    Args:
        prompt: Text prompt for generation
        style_image_paths: List of paths to style images
    
    Returns:
        Generated PIL Image
    """
    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Initialize SDXL pipeline
    print("Loading Stable Diffusion XL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
    )
    pipe = pipe.to(device)
    
    # Load IP-Adapter for SDXL
    print("Loading IP-Adapter...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.safetensors"
    )
    
    # Load style images
    print(f"Loading {len(style_image_paths)} style images...")
    style_images = [load_image(path) for path in style_image_paths]
    
    # Set IP-Adapter scale (can be adjusted)
    pipe.set_ip_adapter_scale(0.7)
    
    # Generate image
    print(f"Generating image with prompt: '{prompt}'...")
    image = pipe(
        prompt=prompt,
        ip_adapter_image=style_images[0],
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]
    
    return image

def main():
    # Load index for retrieval
    embeddings, image_paths = retrieve.load_index()
    
    # Determine device for CLIP
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load CLIP model for search
    model, processor = retrieve.load_clip_model()
    model = model.to(device)
    
    # Search for style images
    query = 'cyberpunk detective'
    print(f"Searching for style images: '{query}'")
    style_image_paths = retrieve.search(
        query, 
        k=3, 
        embeddings=embeddings, 
        image_paths=image_paths,
        model=model,
        processor=processor,
        device=device
    )
    
    print(f"\nUsing {len(style_image_paths)} style images:")
    for i, path in enumerate(style_image_paths, 1):
        print(f"  {i}. {path}")
    
    # Generate image
    prompt = 'cyberpunk detective'
    generated_image = generate_with_style(prompt, style_image_paths)
    
    # Save result
    output_path = "output.png"
    generated_image.save(output_path)
    print(f"\nImage saved to {output_path}")

if __name__ == "__main__":
    main()
