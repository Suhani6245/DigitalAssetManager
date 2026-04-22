import os
import fitz
import docx
from pptx import Presentation
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import re

# -------- MODEL LOADING -------- #
_model = None
_cross_model = None

def get_model():
    global _model
    if _model is None:
        # 🔥 Better model (you can switch back if needed)
        _model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    return _model

def get_cross_encoder():
    global _cross_model
    if _cross_model is None:
        _cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_model


# -------- CLEAN TEXT -------- #
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -------- BETTER CHUNKING (SENTENCE BASED) -------- #
def chunk_text(text, chunk_size=5, overlap=2):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []

    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = " ".join(sentences[i:i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)

    return chunks


# -------- TEXT EXTRACTION -------- #
def extract_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_text(text)


def extract_docx(path):
    doc = docx.Document(path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return clean_text(text)


def extract_pptx(path):
    prs = Presentation(path)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return clean_text(text)


# -------- LOAD DOCUMENTS -------- #
def load_documents(folder):
    docs = []
    names = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if file.endswith(".pdf"):
            text = extract_pdf(path)

        elif file.endswith(".docx"):
            text = extract_docx(path)

        elif file.endswith(".pptx"):
            text = extract_pptx(path)

        else:
            continue

        chunks = chunk_text(text)

        for chunk in chunks:
            docs.append(chunk)
            names.append(file)

    return docs, names


# -------- EMBEDDINGS -------- #
def create_embeddings(docs):
    model = get_model()

    embeddings = model.encode(
        docs,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    faiss.normalize_L2(embeddings)
    return embeddings


# -------- INDEX -------- #
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# -------- SEARCH -------- #
def search(query, docs, names, index, top_k=5):
    model = get_model()
    cross_encoder = get_cross_encoder()

    # 🔥 Add instruction (VERY IMPORTANT for BGE model)
    query = "Represent this sentence for searching relevant documents: " + query

    # -------- STEP 1: RETRIEVAL -------- #
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, top_k * 10)

    candidates = [(names[i], docs[i]) for i in indices[0]]

    # -------- STEP 2: RERANK -------- #
    pairs = [(query, doc[:300]) for _, doc in candidates]

    scores = cross_encoder.predict(pairs)
    scores = np.array(scores)

    reranked = list(zip(candidates, scores))
    reranked.sort(key=lambda x: x[1], reverse=True)

    # -------- REMOVE DUPLICATES -------- #
    results = []
    seen = set()

    for (name, doc), score in reranked:
        if name not in seen:
            results.append((name, doc))
            seen.add(name)

        if len(results) >= top_k:
            break

    return results


# -------- BEST SENTENCE -------- #
def get_best_sentence(text, query):
    model = get_model()

    sentences = re.split(r'(?<=[.!?]) +', text)

    if not sentences:
        return text

    if len(sentences) == 1:
        return sentences[0]

    sent_embeddings = model.encode(sentences, convert_to_numpy=True)
    query_embedding = model.encode([query], convert_to_numpy=True)[0]

    faiss.normalize_L2(sent_embeddings)
    faiss.normalize_L2(query_embedding.reshape(1, -1))

    scores = np.dot(sent_embeddings, query_embedding)

    best_idx = np.argmax(scores)
    return sentences[best_idx]