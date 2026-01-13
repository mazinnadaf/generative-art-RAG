import json
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch

def load_clip_model():
    """Load the OpenAI CLIP model and processor."""
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

def load_index(embeddings_path="style_embeddings.npy", paths_json="image_paths.json"):
    """Load the embeddings and image paths from disk."""
    embeddings = np.load(embeddings_path)
    with open(paths_json, 'r') as f:
        image_paths = json.load(f)
    print(f"Loaded {len(image_paths)} image embeddings.")
    return embeddings, image_paths

def search(query_text, k=3, embeddings=None, image_paths=None, model=None, processor=None, device='cpu'):
    """
    Search for images matching the query text.
    
    Args:
        query_text: Text query to search for
        k: Number of top results to return
        embeddings: Pre-loaded embeddings array
        image_paths: Pre-loaded list of image paths
        model: Pre-loaded CLIP model (optional, will load if not provided)
        processor: Pre-loaded CLIP processor (optional, will load if not provided)
        device: Device to run inference on
    
    Returns:
        List of top k image paths, sorted by similarity (highest first)
    """
    # Load model if not provided
    if model is None or processor is None:
        model, processor = load_clip_model()
    
    model = model.to(device)
    
    # Convert query text to embedding
    inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        # Normalize the embedding
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        query_embedding = text_features.cpu().numpy()
    
    # Compute similarity using dot product (cosine similarity since embeddings are normalized)
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    
    # Get top k indices
    top_k_indices = np.argsort(similarities)[::-1][:k]
    
    # Return corresponding image paths
    results = [image_paths[i] for i in top_k_indices]
    return results

def main():
    # Configuration
    embeddings_path = "style_embeddings.npy"
    paths_json = "image_paths.json"
    
    # Load embeddings and paths
    embeddings, image_paths = load_index(embeddings_path, paths_json)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load CLIP model
    model, processor = load_clip_model()
    model = model.to(device)
    
    # Test search
    query = 'neon rainy street'
    print(f"\nSearching for: '{query}'")
    results = search(query, k=3, embeddings=embeddings, image_paths=image_paths, 
                     model=model, processor=processor, device=device)
    
    print(f"\nFound {len(results)} results:")
    for i, path in enumerate(results, 1):
        print(f"{i}. {path}")

if __name__ == "__main__":
    main()
