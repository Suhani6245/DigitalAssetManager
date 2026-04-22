import streamlit as st
import pickle
import torch
import clip
from PIL import Image
import os

from src.document_search import (
    load_documents,
    create_embeddings,
    build_index,
    search as doc_search,
    get_best_sentence
)

# ---------- LOAD CLIP MODEL ---------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ---------- LOAD IMAGE EMBEDDINGS ---------- #
with open("embeddings/image_embeddings.pkl", "rb") as f:
    embeddings, image_paths = pickle.load(f)

# ---------- IMAGE SEARCH FUNCTION ---------- #
def search_images(query, top_k=5):
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


# ---------- CACHE DOCUMENT SYSTEM ---------- #
@st.cache_resource
def load_doc_system():
    docs, names = load_documents("DataSet/Documents")
    embeddings = create_embeddings(docs)
    index = build_index(embeddings)
    return docs, names, index


# ---------- UI ---------- #
st.set_page_config(page_title="AI Digital Asset Manager", layout="wide")

st.title("AI Digital Asset Manager 🏷️")
st.info("This system uses CLIP for image search and semantic embeddings for document search.")
st.divider()


# ================= IMAGE SEARCH ================= #
st.markdown("### 🔍 Search Images using Natural Language")

query = st.text_input("Enter your image search query")

st.markdown("💡 Try: *dog in water*, *sunset at beach*, *girl walking*")

top_k = st.slider("Number of results", 1, 20, 5)

if query:
    with st.spinner("Searching images..."):
        results = search_images(query, top_k)

    st.success("Image results found!")
    st.divider()

    cols = st.columns(3)

    for i, (img_name, score) in enumerate(results):
        img_path = os.path.join("Images", img_name)

        if os.path.exists(img_path):
            with cols[i % 3]:
                st.image(Image.open(img_path), use_container_width=True)
                st.caption(f"{img_name}")
                st.badge(f"Score: {score:.4f}")


# ================= DOCUMENT SEARCH ================= #
st.divider()
st.markdown("### 📂 Search Documents (PDF, DOCX, PPTX)")

doc_query = st.text_input("Enter your document search query")

if doc_query:
    with st.spinner("Searching documents..."):
        docs, names, index = load_doc_system()
        results = doc_search(doc_query, docs, names, index)

    st.success("Document results found!")
    st.divider()

    for name, preview in results:
        st.subheader(name)

        # Best sentence extraction (accuracy boost)
        best_line = get_best_sentence(preview, doc_query)
        st.write(best_line)