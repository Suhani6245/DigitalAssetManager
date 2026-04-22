import os
import fitz
import docx
from pptx import Presentation
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# -------- MODEL LOADING -------- #
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-mpnet-base-v2')
    return _model


# -------- CLEAN TEXT -------- #
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -------- CHUNKING (IMPROVED) -------- #
def chunk_text(text, chunk_size=120, overlap=40):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 30:  # ignore tiny chunks
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
    embeddings = model.encode(docs)

    embeddings = np.array(embeddings)
    faiss.normalize_L2(embeddings)  # cosine similarity

    return embeddings


# -------- INDEX -------- #
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings)
    return index


# -------- SEARCH -------- #
def search(query, docs, names, index, top_k=5):
    model = get_model()

    query_vec = model.encode([query])
    query_vec = np.array(query_vec)

    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, top_k * 3)  # more candidates

    results = []
    seen = set()

    for i in indices[0]:
        if names[i] not in seen:
            results.append((names[i], docs[i]))
            seen.add(names[i])

        if len(results) >= top_k:
            break

    return results


# -------- BEST SENTENCE (HIGH ACCURACY) -------- #
def get_best_sentence(text, query):
    model = get_model()

    sentences = re.split(r'(?<=[.!?]) +', text)

    if not sentences:
        return text

    if len(sentences) == 1:
        return sentences[0]

    sent_embeddings = model.encode(sentences)
    query_embedding = model.encode([query])[0]

    sent_embeddings = np.array(sent_embeddings)
    query_embedding = np.array(query_embedding)

    # ✅ Normalize for cosine similarity
    faiss.normalize_L2(sent_embeddings)
    faiss.normalize_L2(query_embedding.reshape(1, -1))

    scores = np.dot(sent_embeddings, query_embedding)

    best_idx = np.argmax(scores)
    return sentences[best_idx]