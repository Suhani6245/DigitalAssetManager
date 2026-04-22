import streamlit as st
import pickle
import torch
import clip
from PIL import Image
import os

# ---------- LOAD MODEL ---------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ---------- LOAD EMBEDDINGS ---------- #
with open("embeddings/image_embeddings.pkl", "rb") as f:
    embeddings, image_paths = pickle.load(f)

# ---------- SEARCH FUNCTION ---------- #
def search(query, top_k=5):
    text = clip.tokenize([query]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text)
    
    results = []
    
    for img_name, img_embed in embeddings.items():
        img_embed = torch.tensor(img_embed).to(device)
        
        similarity = torch.cosine_similarity(text_features, img_embed).item()
        results.append((img_name, similarity))
    
    results = sorted(results, key=lambda x: x[1], reverse=True)
    
    return results[:top_k]

# ---------- UI ---------- #
st.set_page_config(page_title="AI Digital Asset Manager", layout="wide")

st.title("AI Digital Asset Manager🏷️")
st.info("This system uses CLIP to match text and images using semantic similarity instead of keywords.")
st.divider()
st.markdown("### Search images using natural language")


# ---------- INPUT ---------- #
query = st.text_input("Enter your search query")

st.markdown("💡 Try: *dog in water*, *sunset at beach*, *girl walking*")

top_k = st.slider("Number of results", 1, 20, 5)

# ---------- SEARCH ---------- #
if query:
    with st.spinner("Searching..."):
        results = search(query, top_k)

    st.success("Results found!")
    st.divider()
    # ---------- DISPLAY RESULTS ---------- #
    cols = st.columns(3)

    for i, (img_name, score) in enumerate(results):
        img_path = os.path.join("images", img_name)
        
        if os.path.exists(img_path):
            with cols[i % 3]:
                st.image(Image.open(img_path), use_container_width=True)
                st.caption(f"{img_name}")
                st.badge(f"Score: {score:.4f}")



