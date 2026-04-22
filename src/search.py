import torch
import clip
import pickle
import numpy as np

# -------- LOAD MODEL -------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

# -------- LOAD EMBEDDINGS -------- #
with open("embeddings/image_embeddings.pkl", "rb") as f:
    embeddings, image_paths = pickle.load(f)

# -------- SEARCH FUNCTION -------- #
def search(query, top_k=5):
    text = clip.tokenize([query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scores = []

    for img_name in image_paths:
        img_feat = embeddings[img_name]
        score = np.dot(text_features.cpu().numpy(), img_feat.T)
        scores.append((img_name, score[0][0]))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_k]

# -------- TEST -------- #
if __name__ == "__main__":
    query = input("🔍 Enter search query: ")

    results = search(query)

    print("\nTop Results:\n")
    for img, score in results:
        print(f"{img}  -->  {score:.4f}")