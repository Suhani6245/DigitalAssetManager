import streamlit as st
import pickle
import torch
import clip
from PIL import Image
import os
import numpy as np

from src.document_search import (
    load_documents,
    create_embeddings,
    build_index,
    search as doc_search,
    get_best_sentence
)

# ---------- DEVICE ---------- #
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- LOAD CLIP MODEL (CACHED) ---------- #
@st.cache_resource
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

model, preprocess = load_clip_model()

# ---------- LOAD IMAGE EMBEDDINGS ---------- #
@st.cache_resource
def load_image_embeddings():
    with open("embeddings/image_embeddings.pkl", "rb") as f:
        data = pickle.load(f)

    # CASE 1: Already correct (dict)
    if isinstance(data, dict):
        return data

    # CASE 2: Tuple/list (embeddings, paths)
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        embeddings, paths = data

        # If embeddings is list/array → normal case
        if isinstance(embeddings, (list, np.ndarray)):
            return {paths[i]: embeddings[i] for i in range(len(paths))}

        # If embeddings is dict → already mapped
        elif isinstance(embeddings, dict):
            return embeddings

    # CASE 3: Unknown format → debug
    print("DEBUG TYPE:", type(data))
    raise ValueError("Unsupported embedding format")

image_embeddings = load_image_embeddings()



# ---------- IMAGE SEARCH FUNCTION ---------- #
def search_images(query, top_k=5):
    text = clip.tokenize([query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results = []

    for img_name, img_embed in image_embeddings.items():
        img_embed = torch.tensor(img_embed).to(device)
        img_embed = img_embed / img_embed.norm()

        similarity = torch.cosine_similarity(text_features, img_embed).item()
        results.append((img_name, similarity))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[:top_k]

# ---------- DOCUMENT SYSTEM (CACHED) ---------- #
@st.cache_resource
def load_doc_system():
    docs, names = load_documents("DataSet/Documents")
    embeddings = create_embeddings(docs)
    index = build_index(embeddings)
    return docs, names, index

docs, names, index = load_doc_system()

# ---------- UI ---------- #
st.set_page_config(page_title="AI Digital Asset Manager", layout="wide")

st.title("AI Digital Asset Manager 🏷️")
st.info("Search images and documents using natural language (multimodal search).")
st.divider()

# ---------- INPUT ---------- #
query = st.text_input("🔍 Search your files (images + documents)")
top_k = st.slider("Number of results", 1, 20, 5)
filter_type = st.radio("Filter results", ["All", "Images", "Documents"])

# ---------- SEARCH ---------- #
if query:

    image_results = []
    doc_results = []

    # -------- IMAGE SEARCH -------- #
    if filter_type in ["All", "Images"]:
        with st.spinner("Searching images..."):
            image_results = search_images(query, top_k)

    # -------- DOCUMENT SEARCH -------- #
    if filter_type in ["All", "Documents"]:
        with st.spinner("Searching documents..."):
            doc_results = doc_search(query, docs, names, index, top_k)

    st.success("Results found!")
    st.divider()

    # ---------- TABS ---------- #
    tab1, tab2 = st.tabs(["🖼️ Images", "📄 Documents"])

    # ================= IMAGES ================= #
    with tab1:
        if image_results:
            cols = st.columns(3)

            for i, (img_name, score) in enumerate(image_results):
                img_path = os.path.join("DataSet/Images", img_name)

                if os.path.exists(img_path):
                    with cols[i % 3]:
                        st.image(Image.open(img_path), use_container_width=True)
                        st.caption(img_name)
                        st.badge(f"Score: {score:.4f}")
        else:
            st.warning("No image results found.")

    # ================= DOCUMENTS ================= #
    with tab2:
        if doc_results:

            # ---------- HELPERS ---------- #
            def highlight_text(text, query):
                if not query:
                    return text
                return text.replace(query, f"**{query}**")

            def get_preview_window(text, best_sentence, window=200):
                idx = text.find(best_sentence)

                if idx == -1:
                    return text[:500]

                start = max(0, idx - window)
                end = min(len(text), idx + len(best_sentence) + window)

                return text[start:end]

            # ---------- DISPLAY ---------- #
            for name, preview in doc_results:

                best_line = get_best_sentence(preview, query)
                preview_window = get_preview_window(preview, best_line)

                highlighted_preview = highlight_text(preview_window, query)
                highlighted_best = highlight_text(best_line, query)

                file_path = os.path.join("DataSet/Documents", name)

                with st.expander(f"📄 {name}"):

                    st.markdown(f"**🔍 Best Match:** {highlighted_best}")

                    st.markdown("**📄 Context Preview:**")
                    st.markdown(highlighted_preview + "...")

                    st.divider()

                    # ---------- OPEN / DOWNLOAD BUTTON ---------- #
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label="📂 Open / Download Document",
                                data=f,
                                file_name=name,
                                mime="application/octet-stream"
                            )
                    else:
                        st.error("File not found.")

    # ---------- EMPTY CASE ---------- #
    if not image_results and not doc_results:
        st.error("No results found. Try a different query.")