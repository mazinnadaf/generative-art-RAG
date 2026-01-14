# Cyberpunk RAG Image Generator

A full-stack AI image generation app that uses **Retrieval-Augmented Generation (RAG)** to style-condition Stable Diffusion.

Unlike standard generators that just take a prompt, this system first searches a vector database of curated cyberpunk artwork to find the perfect **visual style match** for your idea, then uses that reference image to guide the generation process.

### https://frontend-hazel-rho-53.vercel.app/

---

## Features
* **Style RAG:** Retrieves the most relevant style reference from a vector index of cyberpunk artworks based on your prompt.
* **SDXL + IP-Adapter:** Uses Stable Diffusion XL combined with IP-Adapter for high-fidelity style transfer.
* **Serverless GPU Backend:** Powered by **Modal** (running on A10G GPUs).
* **Modern Frontend:** Built with React, TypeScript, Tailwind CSS, and Vite.
* **Real-time Feedback:** Shows the actual reference image used for generation alongside the result.

## Tech Stack
* **Frontend:** React, Tailwind CSS, Vercel
* **Backend:** Python, Modal (Serverless), FastAPI
* **AI Models:**
    * Stable Diffusion XL (Base Model)
    * CLIP (Semantic Search & Embeddings)
    * IP-Adapter (Image Prompt Adapter)

## How It Works
1.  **User Input:** You type a prompt (e.g., "Neon rainy street").
2.  **Vector Search:** The backend embeds your text using CLIP and searches the pre-computed index of cyberpunk art styles.
3.  **Retrieval:** The system picks the closest matching artistic style (e.g., a specific color palette or composition).
4.  **Generation:** The retrieved image is fed into SDXL via IP-Adapter to generate your unique image using that specific style.

---

### Running Locally

```bash
// Backend
modal serve main_modal.py

//Frontend
cd frontend
npm run dev
