import streamlit as st
import pickle
import torch
import clip
from PIL import Image
import os
import numpy as np
import time

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
st.markdown("""<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

body {
    background: linear-gradient(135deg, #f8fafc, #e2e8f0);
}

@media (prefers-color-scheme: dark) {
    body {
        background: linear-gradient(135deg, #0f172a, #020617);
        color: #e2e8f0;
    }
}

.hero { text-align: center; padding: 10px 0 20px 0; }
.hero h1 { font-size: 40px; font-weight: 800; }
.hero p { color: gray; font-size: 14px; }

.glass {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

.doc-card {
    background: rgba(255,255,255,0.75);
    padding: 15px;
    border-radius: 12px;
}

section[data-testid="stSidebar"] {
    background: #111827;
}
section[data-testid="stSidebar"] * {
    color: white;
}

.center {
    text-align: center;
    color: gray;
    margin-top: 60px;
}
</style>""", unsafe_allow_html=True)

# ---------- HERO ---------- #
st.markdown("""
<div class="hero">
<h1>🧠 AI Digital Asset Manager</h1>
<p>Search images & documents using natural language</p>
</div>
""", unsafe_allow_html=True)

# ---------- DEVICE ---------- #
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- LOAD CLIP ---------- #
@st.cache_resource
def load_clip_model():
    return clip.load("ViT-B/32", device=device)

model, preprocess = load_clip_model()

# ---------- LOAD IMAGE EMBEDDINGS (OPTIMIZED) ---------- #
@st.cache_resource
def load_image_embeddings():
    with open("embeddings/image_embeddings.pkl", "rb") as f:
        data = pickle.load(f)

    if isinstance(data, tuple):
        data = data[0]

    names, captions, matrix = [], [], []

    for img_name, val in data.items():
        emb = np.array(val["embedding"]).squeeze()

        if emb.ndim != 1:
            continue

        names.append(img_name)
        captions.append(val.get("caption", ""))
        matrix.append(emb)

    matrix = np.vstack(matrix).astype("float32")

    # 🔥 normalize once globally
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    return names, captions, matrix


img_names, img_captions, img_matrix = load_image_embeddings()

# ---------- IMAGE SEARCH (FAST) ---------- #
def search_images(query, top_k=5):
    text = clip.tokenize([query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text).cpu().numpy()

    text_features /= np.linalg.norm(text_features)

    # 🔥 vectorized similarity
    scores = np.dot(img_matrix, text_features.T).squeeze()

    top_idx = np.argsort(-scores)[:top_k]

    return [
        (img_names[i], float(scores[i]), img_captions[i])
        for i in top_idx
    ]

# ---------- FILENAME SEARCH ---------- #
def search_images_by_filename(query, top_k=5):
    query_lower = query.lower()
    results = []

    for img_name in img_names:
        if query_lower in os.path.basename(img_name).lower():
            results.append((img_name, 1.0, ""))

    return results[:top_k]

# ---------- DOCUMENT SYSTEM ---------- #
@st.cache_resource
def load_doc_system():
    docs, names = load_documents("DataSet/Documents")

    embeddings = create_embeddings(docs)
    index = build_index(embeddings)

    return docs, names, index

docs, names, index = load_doc_system()

# ---------- DOC FILENAME SEARCH ---------- #
def search_docs_by_filename(query, docs, names, top_k=5):
    query_lower = query.lower()
    results = []
    seen = set()

    for name, doc in zip(names, docs):
        if query_lower in name.lower() and name not in seen:
            results.append((name, doc))
            seen.add(name)

        if len(results) >= top_k:
            break

    return results

# ---------- SIDEBAR ---------- #
st.sidebar.title("⚙️ Controls")

query = st.sidebar.text_input("🔍 Search")
top_k = st.sidebar.slider("Results", 1, 20, 5)
filter_type = st.sidebar.radio("Filter", ["All", "Images", "Documents"])
search_by_filename = st.sidebar.checkbox("Search by filename only", value=False)

st.sidebar.markdown("---")
st.sidebar.info("💡 Try:\n- sunset beach\n- AI notes\n- dog running")

# ---------- DOWNLOAD ---------- #
st.sidebar.markdown("---")
pdf_path = "AI_DAM_Documentation.pdf"

if os.path.exists(pdf_path):
    with open(pdf_path, "rb") as f:
        st.sidebar.download_button(
            "📥 Download Project Docs",
            data=f,
            file_name="AI_DAM_Documentation.pdf"
        )

# ---------- EMPTY ---------- #
if not query:
    st.markdown("<div class='center'><h3>Start searching 🔍</h3></div>", unsafe_allow_html=True)
    st.stop()

# ---------- SEARCH ---------- #
image_results, doc_results = [], []
img_elapsed, doc_elapsed = 0, 0

if filter_type in ["All", "Images"]:
    with st.spinner("Searching images…"):
        start = time.time()

        image_results = (
            search_images_by_filename(query, top_k)
            if search_by_filename else
            search_images(query, top_k)
        )

        img_elapsed = time.time() - start

if filter_type in ["All", "Documents"]:
    with st.spinner("Searching & reranking documents…"):
        start = time.time()

        doc_results = (
            search_docs_by_filename(query, docs, names, top_k)
            if search_by_filename else
            doc_search(query, docs, names, index, top_k)
        )

        doc_elapsed = time.time() - start

# ---------- RESULT COUNT ---------- #
st.success(f"{len(image_results) + len(doc_results)} results found")

# ---------- TABS ---------- #
tab1, tab2 = st.tabs(["🖼️ Images", "📄 Documents"])

# ---------- IMAGES ---------- #
with tab1:
    st.markdown(f"#### 🖼 Image Results `{len(image_results)} found`")
    st.caption(f"{img_elapsed:.2f}s")

    if image_results:
        cols = st.columns(3)

        for i, (img_key, score, caption) in enumerate(image_results):
            img_path = os.path.join("DataSet/Images", os.path.basename(img_key))

            if os.path.exists(img_path):
                with cols[i % 3]:
                    st.markdown('<div class="glass">', unsafe_allow_html=True)
                    st.image(Image.open(img_path), width="stretch")
                    st.markdown(f"**{os.path.basename(img_key)}**")
                    st.caption(caption)
                    st.caption(f"Similarity: {score:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"Missing: {img_path}")
    else:
        st.warning("No image results found.")

# ---------- DOCUMENTS ---------- #
with tab2:
    st.markdown(f"#### 📄 Document Results `{len(doc_results)} found`")
    st.caption(f"{doc_elapsed:.2f}s")

    if doc_results:
        for name, preview, *_ in doc_results:
            best_line = get_best_sentence(preview, query)

            file_path = os.path.join("DataSet/Documents", name)

            with st.expander(f"📄 {name}"):
                st.markdown(f"""
                <div class="doc-card">
                <b>🔍 Best Match:</b><br>{best_line}<br><br>
                <b>📄 Preview:</b><br>{preview[:400]}...
                </div>
                """, unsafe_allow_html=True)

                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        st.download_button("📂 Open / Download", data=f, file_name=name)
    else:
        st.warning("No document results found.")