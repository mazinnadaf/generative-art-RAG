import requests
import base64
import json

url = "https://mazinnadaf--rag-image-generator-imagegenerator-generate-dev.modal.run"

prompt = input("Enter a prompt: ")
print(f"Requesting image for: '{prompt}'...")

response = requests.post(url, json={"prompt": prompt})

if response.status_code == 200:
    data = response.json()
    
    image_data = data["image"].split(",")[1]
    filename = f"{prompt.replace(' ', '_')}.png"
    
    with open(filename, "wb") as f:
        f.write(base64.b64decode(image_data))
        
    print(f"Saved to {filename}")
    print(f"   (Style reference used: {data.get('reference_image')})")
else:
    print(f"Error {response.status_code}: {response.text}")