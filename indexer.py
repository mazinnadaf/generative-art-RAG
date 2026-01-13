import os
import json
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

def load_clip_model():
    """Load the OpenAI CLIP model and processor."""
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

def scan_image_files(directory):
    """Scan directory for image files (.jpg, .jpeg, .png, .webp)."""
    image_files = []
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist.")
        return image_files
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.webp')
    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_formats):
            image_files.append(os.path.join(directory, filename))
    
    image_files.sort()
    print(f"Found {len(image_files)} image files.")
    return image_files

def compute_embeddings(model, processor, image_paths, device='cpu'):
    """Compute embeddings for a list of images."""
    embeddings = []
    valid_paths = []
    
    print("Computing embeddings...")
    model = model.to(device)
    
    for i, image_path in enumerate(image_paths):
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Compute embedding
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.cpu().numpy())
                valid_paths.append(image_path)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images...")
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    if embeddings:
        embeddings_array = np.vstack(embeddings).astype('float32')
        print(f"Computed embeddings for {len(valid_paths)} images.")
        return embeddings_array, valid_paths
    else:
        print("No valid embeddings computed.")
        return None, []

def save_embeddings(embeddings, npy_path):
    """Save embeddings to a NumPy file."""
    np.save(npy_path, embeddings)
    print(f"Saved embeddings to {npy_path} (shape: {embeddings.shape})")

def save_image_paths(image_paths, json_path):
    """Save list of image paths to a JSON file."""
    with open(json_path, 'w') as f:
        json.dump(image_paths, f, indent=2)
    print(f"Saved {len(image_paths)} image paths to {json_path}")

def main():
    # Configuration
    dataset_dir = "./style_dataset"
    embeddings_path = "style_embeddings.npy"
    paths_json = "image_paths.json"
    
    # Load CLIP model
    model, processor = load_clip_model()
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Scan for image files
    image_paths = scan_image_files(dataset_dir)
    
    if not image_paths:
        print("No image files found. Exiting.")
        return
    
    # Compute embeddings
    embeddings, valid_paths = compute_embeddings(model, processor, image_paths, device)
    
    if embeddings is None or len(valid_paths) == 0:
        print("No embeddings to save. Exiting.")
        return
    
    # Save embeddings as NumPy file
    save_embeddings(embeddings, embeddings_path)
    
    # Save image paths
    save_image_paths(valid_paths, paths_json)
    
    print("Indexing complete!")

if __name__ == "__main__":
    main()
