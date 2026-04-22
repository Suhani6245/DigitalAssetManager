import os
import fitz
import docx
from pptx import Presentation
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# ✅ Best model for semantic search
model = SentenceTransformer('all-mpnet-base-v2')


# -------- CLEAN TEXT -------- #
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -------- CHUNKING -------- #
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
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
    embeddings = model.encode(docs)
    embeddings = np.array(embeddings)

    # ✅ Normalize (important for cosine similarity)
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
    query_vec = model.encode([query])
    query_vec = np.array(query_vec)

    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, top_k * 2)

    results = []
    seen = set()

    for i in indices[0]:
        if names[i] not in seen:
            results.append((names[i], docs[i]))
            seen.add(names[i])

        if len(results) >= top_k:
            break

    return results


# -------- BEST SENTENCE (🔥 ACCURACY BOOST) -------- #
def get_best_sentence(text, query):
    sentences = re.split(r'(?<=[.!?]) +', text)

    if not sentences:
        return text

    sent_embeddings = model.encode(sentences)
    query_embedding = model.encode([query])[0]

    scores = np.dot(sent_embeddings, query_embedding)

    best_idx = np.argmax(scores)
    return sentences[best_idx]