import os
import torch
import clip
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# =========================
# CONFIG
# =========================
IMAGE_FOLDER = "DataSet/Images"
OUTPUT_FILE = "embeddings/image_embeddings.pkl"

# Reduced batch size for CPU to prevent RAM overload and lagging!
BATCH_SIZE = 4   
SAVE_EVERY = 200  # checkpoint saving

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    # Optimizes CPU usage to prevent freezing
    torch.set_num_threads(min(8, os.cpu_count() or 4))

# =========================
# LOAD MODELS
# =========================
print("Loading models...")

clip_model, preprocess = clip.load("ViT-B/32", device=device)

blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

clip_model.eval()
blip_model.eval()

# =========================
# LOAD EXISTING EMBEDDINGS
# =========================
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "rb") as f:
        image_db = pickle.load(f)
else:
    image_db = {}

# =========================
# GET IMAGE PATHS
# =========================
all_images = [
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

# 🔥 only new images
new_images = [
    img for img in all_images
    if img not in image_db
]

print(f"Total images: {len(all_images)}")
print(f"New images to process: {len(new_images)}")

# =========================
# PROCESS IN BATCHES
# =========================
for i in tqdm(range(0, len(new_images), BATCH_SIZE)):
    batch_paths = new_images[i:i + BATCH_SIZE]

    images = []
    valid_paths = []

    # -------- LOAD IMAGES -------- #
    for path in batch_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_paths.append(path)
        except:
            continue

    if not images:
        continue

    # -------- BLIP CAPTIONING (BATCHED) -------- #
    with torch.no_grad():
        inputs = blip_processor(
            images=images,
            return_tensors="pt"
        ).to(device)

        outputs = blip_model.generate(
            **inputs,
            max_length=30
        )

        captions = blip_processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )

    # -------- CLIP IMAGE EMBEDDINGS -------- #
    image_tensors = torch.stack([
        preprocess(img) for img in images
    ]).to(device)

    with torch.no_grad():
        img_features = clip_model.encode_image(image_tensors)

    img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    # -------- CLIP TEXT EMBEDDINGS (CAPTIONS) -------- #
    # truncate=True prevents crashes if BLIP generates a caption > 77 tokens
    try:
        text_tokens = clip.tokenize(captions, truncate=True).to(device)
    except TypeError:
        # Fallback if the clip version doesn't support truncate=True
        text_tokens = clip.tokenize([c[:200] for c in captions]).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # -------- HYBRID EMBEDDING -------- #
    hybrid_features = (img_features + text_features) / 2.0
    hybrid_features = hybrid_features / hybrid_features.norm(dim=-1, keepdim=True)

    hybrid_features = hybrid_features.cpu().numpy()

    # -------- STORE -------- #
    for j, path in enumerate(valid_paths):
        image_db[path] = {
            "embedding": hybrid_features[j],
            "caption": captions[j]
        }

    # -------- PERIODIC SAVE -------- #
    if (i // BATCH_SIZE) % (SAVE_EVERY // BATCH_SIZE) == 0:
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump(image_db, f)

# =========================
# FINAL SAVE
# =========================
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(image_db, f)

print("✅ Encoding complete!")