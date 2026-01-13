import base64
import io
import os
import sys
import modal
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image

# Config
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHTS = "ip-adapter_sdxl.safetensors"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

LOCAL_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_APP_DIR = "/root"
EMBEDDINGS_PATH = f"{REMOTE_APP_DIR}/style_embeddings.npy"
PATHS_JSON = f"{REMOTE_APP_DIR}/image_paths.json"

# Model Downloader
def download_models():
    from transformers import CLIPModel, CLIPProcessor
    try:
        CLIPModel.from_pretrained(CLIP_MODEL_ID, use_safetensors=True)
    except Exception:
        CLIPModel.from_pretrained(CLIP_MODEL_ID)
    CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHTS)

# Modal Image Setup
image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi[standard]", 
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "numpy",
        "safetensors",
        "pillow",
    )
    .run_function(download_models)
    .add_local_file(os.path.join(LOCAL_PROJECT_DIR, "retrieve.py"), remote_path=f"{REMOTE_APP_DIR}/retrieve.py")
    .add_local_file(os.path.join(LOCAL_PROJECT_DIR, "style_embeddings.npy"), remote_path=EMBEDDINGS_PATH)
    .add_local_file(os.path.join(LOCAL_PROJECT_DIR, "image_paths.json"), remote_path=PATHS_JSON)
    .add_local_dir(os.path.join(LOCAL_PROJECT_DIR, "style_dataset"), remote_path=f"{REMOTE_APP_DIR}/style_dataset")
)

app = modal.App("rag-image-generator")

# App Class
@app.cls(gpu="A10G", image=image)
class ImageGenerator:
    @modal.enter()
    def setup(self):
        print("Container starting... Loading models and index.")
        import retrieve
        self.retrieve = retrieve
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load index
        self.embeddings, self.image_paths = self.retrieve.load_index(
            embeddings_path=EMBEDDINGS_PATH, paths_json=PATHS_JSON
        )
        
        # Load CLIP
        self.clip_model, self.clip_processor = self.retrieve.load_clip_model()
        self.clip_model = self.clip_model.to(self.device)

        # Load Stable Diffusion
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
        # Load IP Adapter
        self.pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHTS)
        self.pipe.set_ip_adapter_scale(0.7)
        print("Setup complete.")

    @modal.asgi_app()
    def generate(self):
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        import base64
        import io

        web_app = FastAPI()

        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.post("/")
        async def generate_image(body: dict):
            prompt = body.get("prompt")
            if not prompt:
                return {"error": "Prompt is required."}

            print(f"Searching for: {prompt}")
            
            # Retrieve Style
            results = self.retrieve.search(
                prompt,
                k=1,
                embeddings=self.embeddings,
                image_paths=self.image_paths,
                model=self.clip_model,
                processor=self.clip_processor,
                device=self.device,
            )
            
            if not results:
                return {"error": "No reference images available."}

            style_path = self._resolve_path(results[0])
            print(f"Using style: {style_path}")
            # Load image as PIL object
            style_image_pil = load_image(style_path)

            # Encode the reference image to Base64 
            ref_buffer = io.BytesIO()
            # Save as PNG to ensure compatibility, even if source is JPG/WebP
            style_image_pil.save(ref_buffer, format="PNG")
            ref_encoded = base64.b64encode(ref_buffer.getvalue()).decode("utf-8")
            ref_data_uri = f"data:image/png;base64,{ref_encoded}"
            # -------------------------------------------------

            # Generate Image (using the PIL object)
            output = self.pipe(
                prompt=prompt,
                ip_adapter_image=[style_image_pil],
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

            # Encode generated image to Base64
            buffer = io.BytesIO()
            output.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Send the data URI instead of the file path
            return {"image": f"data:image/png;base64,{encoded}", "reference_image": ref_data_uri}

        return web_app

    def _resolve_path(self, path: str) -> str:
        if path.startswith("./"):
            path = path[2:]
        if not os.path.isabs(path):
            path = os.path.join(REMOTE_APP_DIR, path)
        return path