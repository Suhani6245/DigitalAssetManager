import os
import torch
import clip
import pickle
from PIL import Image
from tqdm import tqdm

# -------- SETUP -------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_folder = "Images"

#for sample images folder
#image_folder = "Images/sample"

embeddings = {}
image_paths = []

# -------- ENCODING -------- #
for img_name in tqdm(os.listdir(image_folder), desc="Encoding Images"):
    path = os.path.join(image_folder, img_name)

    try:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model.encode_image(image)

        feature = feature / feature.norm(dim=-1, keepdim=True)

        embeddings[img_name] = feature.cpu().numpy()
        image_paths.append(img_name)

    except Exception as e:
        print(f"Skipping {img_name}: {e}")
        continue

# -------- SAVE -------- #
os.makedirs("embeddings", exist_ok=True)

with open("embeddings/image_embeddings.pkl", "wb") as f:
    pickle.dump((embeddings, image_paths), f)

print("✅ Image embeddings saved successfully!")