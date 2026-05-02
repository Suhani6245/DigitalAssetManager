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

# ---------- PAGE CONFIG ---------- #
st.set_page_config(page_title="AI DAM", layout="wide")

# ---------- PREMIUM UI CSS ---------- #
st.markdown("""
<style>

/* -------- GLOBAL -------- */
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Light Theme */
body {
    background: linear-gradient(135deg, #f8fafc, #e2e8f0);
}

/* Dark Mode */
@media (prefers-color-scheme: dark) {
    body {
        background: linear-gradient(135deg, #0f172a, #020617);
        color: #e2e8f0;
    }
}

/* -------- HERO -------- */
.hero {
    text-align: center;
    padding: 10px 0 20px 0;
}

.hero h1 {
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero p {
    color: gray;
    font-size: 14px;
}

/* -------- GLASS CARD -------- */
.glass {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    transition: 0.2s;
}

@media (prefers-color-scheme: dark) {
    .glass {
        background: rgba(30,41,59,0.6);
    }
}

.glass:hover {
    transform: translateY(-4px);
}

/* -------- DOC BOX -------- */
.doc-card {
    background: rgba(255,255,255,0.75);
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
    line-height: 1.6;
}

@media (prefers-color-scheme: dark) {
    .doc-card {
        background: rgba(30,41,59,0.6);
    }
}

/* -------- SIDEBAR -------- */
section[data-testid="stSidebar"] {
    background: #111827;
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* -------- BUTTON -------- */
.stDownloadButton button {
    width: 100%;
    border-radius: 10px;
}

/* -------- EMPTY STATE -------- */
.center {
    text-align: center;
    color: gray;
    margin-top: 60px;
}

</style>
""", unsafe_allow_html=True)

# ---------- HERO ---------- #
st.markdown("""
<div class="hero">
    <h1>🧠 AI Digital Asset Manager</h1>
    <p>Search images & documents using natural language</p>
</div>
""", unsafe_allow_html=True)

# ---------- DEVICE ---------- #
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- LOAD MODELS ---------- #
@st.cache_resource
def load_clip_model():
    return clip.load("ViT-B/32", device=device)

model, preprocess = load_clip_model()

# ---------- LOAD IMAGE EMBEDDINGS ---------- #
@st.cache_resource
def load_image_embeddings():
    with open("embeddings/image_embeddings.pkl", "rb") as f:
        data = pickle.load(f)

    if isinstance(data, tuple) and len(data) > 0:
        data = data[0]

    if isinstance(data, dict):
        return data

    # Fallback for other list/tuple formats
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        embeddings, paths = data
        return {paths[i]: embeddings[i] for i in range(len(paths))}

    raise ValueError("Invalid embedding format")

image_embeddings = load_image_embeddings()

# ---------- IMAGE SEARCH ---------- #
def search_images(query, top_k=5):
    text = clip.tokenize([query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    results = []

    for img_name, img_embed in image_embeddings.items():
        img_embed = torch.tensor(img_embed).to(device)
        img_embed /= img_embed.norm()

        score = torch.cosine_similarity(text_features, img_embed).item()
        results.append((img_name, score))

    return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

def search_images_by_filename(query, top_k=5):
    query_lower = query.lower()
    results = []
    
    for img_name in image_embeddings.keys():
        file_name = os.path.basename(img_name)
        if query_lower in file_name.lower():
            # Give it a dummy score of 1.0 since it's an exact filename match
            results.append((img_name, 1.0))
            
    return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

# ---------- DOCUMENT SYSTEM ---------- #
@st.cache_resource
def load_doc_system():
    docs, names = load_documents("DataSet/Documents")
    embeddings = create_embeddings(docs)
    index = build_index(embeddings)
    return docs, names, index

docs, names, index = load_doc_system()

def search_docs_by_filename(query, docs, names, top_k=5):
    query_lower = query.lower()
    results = []
    seen = set()
    
    for name, doc in zip(names, docs):
        if query_lower in name.lower():
            if name not in seen:
                results.append((name, doc))
                seen.add(name)
            if len(results) >= top_k:
                break
                
    return results

# ---------- SIDEBAR ---------- #
st.sidebar.title("⚙️ Controls")

# 1. Search Query
query = st.sidebar.text_input("🔍 Search")

# 2. Results Slider
top_k = st.sidebar.slider("Results", 1, 20, 5)

# 3. Filter Type
filter_type = st.sidebar.radio("Filter", ["All", "Images", "Documents"])

# 4. Search by Filename
search_by_filename = st.sidebar.checkbox("Search by filename only", value=False)

st.sidebar.markdown("---")

# 4. Suggestions
st.sidebar.info("💡 Try:\n- sunset beach\n- AI notes\n- dog running")


# ---------- SIDEBAR FOOTER / DOWNLOAD ---------- #
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Documentation")

pdf_path = "AI_DAM_Documentation.pdf"

if os.path.exists(pdf_path):
    with open(pdf_path, "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Project Docs",
            data=f,
            file_name="AI_DAM_Documentation.pdf",
            mime="application/pdf",
            width="stretch"
        )
else:
    st.sidebar.warning("Documentation file not found.")

if not query:
    st.markdown("""
    <div class="center">
        <h3>Start searching 🔍</h3>
        <p>Try something like <i>“sunset beach”</i></p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

image_results = []
doc_results = []

if filter_type in ["All", "Images"]:
    with st.spinner("Searching images..."):
        if search_by_filename:
            image_results = search_images_by_filename(query, top_k)
        else:
            image_results = search_images(query, top_k)

if filter_type in ["All", "Documents"]:
    with st.spinner("Searching documents..."):
        if search_by_filename:
            doc_results = search_docs_by_filename(query, docs, names, top_k)
        else:
            doc_results = doc_search(query, docs, names, index, top_k)

st.success(f"{len(image_results) + len(doc_results)} results found")

tab1, tab2 = st.tabs(["🖼️ Images", "📄 Documents"])



# ---------- IMAGES ---------- #
with tab1:
    if image_results:
        cols = st.columns(3)
        for i, (img_key, score) in enumerate(image_results):
            file_name = os.path.basename(img_key)
            img_path = os.path.join("DataSet/Images", file_name)
            if os.path.exists(img_path):
                with cols[i % 3]:
                    st.markdown('<div class="glass">', unsafe_allow_html=True)
                    st.image(Image.open(img_path), width="stretch")
                    st.markdown(f"**{file_name}**")
                    st.caption(f"Similarity: {score:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Debug message to show exactly what's failing
                st.error(f"Failed to find: {img_path}")
    else:
        st.warning("No image results found.")

# ---------- DOCUMENTS ---------- #
with tab2:
    if doc_results:
        for name, preview in doc_results:
            best_line = get_best_sentence(preview, query)
            file_path = os.path.join("DataSet/Documents", name)
            with st.expander(f"📄 {name}"):
                st.markdown(f"""
                <div class="doc-card">
                <b>🔍 Best Match:</b><br>
                {best_line}
                <br><br>
                <b>📄 Preview:</b><br>
                {preview[:400]}...
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        st.download_button("📂 Open / Download", data=f, file_name=name)
    else:
        st.warning("No document results found.")