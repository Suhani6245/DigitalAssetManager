# ⬡ AI Digital Asset Manager

> **Multimodal semantic search** for your local images and documents — powered by BLIP, CLIP, Sentence-Transformers, FAISS, and a CrossEncoder reranker.
> Everything runs **100% locally** — no paid APIs, no cloud, no GPU required.

---

## What's New (v2)

| Update | Detail |
|---|---|
| BLIP image captioning | Captions auto-generated for every image using `Salesforce/blip-image-captioning-base` |
| Hybrid image embeddings | CLIP visual + CLIP text (caption) embeddings averaged into one vector |
| Incremental encoding | Only new images encoded on each run — existing cache preserved |
| Batched BLIP inference | Images processed in configurable batches (default: 4) for CPU safety |
| Periodic checkpointing | Embeddings saved every 200 images to prevent data loss on interruption |
| BLIP memory cleanup | BLIP unloaded from RAM after encoding to free ~900 MB |
| Vectorized image search | NumPy matrix dot-product replaces per-image loop — significantly faster |
| Unified doc + image search | Image captions injected into FAISS document index for cross-modal retrieval |
| Search latency display | Time taken shown in UI after every search |
| Analytics dashboard | Dedicated `analytics.py` page for system and dataset insights |

---

## Features

| Capability | Implementation |
|---|---|
| Image captioning | BLIP (`Salesforce/blip-image-captioning-base`) |
| Image semantic search | CLIP ViT-B/32 + BLIP caption hybrid embedding |
| Document parsing | PyMuPDF · python-docx · python-pptx |
| Document embedding | Sentence Transformers (`BAAI/bge-base-en-v1.5`) |
| Fast candidate retrieval | FAISS `IndexFlatIP` (cosine similarity) |
| Result reranking | CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| Best-sentence extraction | Sentence-level cosine similarity |
| Document preview | Highlighted matching sentence + context |
| Search by filename | Toggle for fast substring matching |
| Image embedding cache | Pickle file — incremental, only new images re-encoded |
| BLIP memory management | Auto-unloaded after indexing |
| Search timing | Displayed in UI after every search |
| Analytics dashboard | Dataset stats, model info, system metrics |
| Fully offline | No external APIs or network calls at search time |

---

## Tech Stack

- **Python 3.9+**
- **Streamlit** — user interface and analytics dashboard
- **OpenAI CLIP** — image-text alignment and visual encoding
- **BLIP** (Salesforce, via HuggingFace Transformers) — automatic image captioning
- **Sentence Transformers** — document embeddings and CrossEncoder reranking
- **FAISS** — approximate nearest-neighbour vector search
- **PyMuPDF** — PDF text extraction
- **python-docx** — DOCX parsing
- **python-pptx** — PPTX parsing

---

## Project Structure

```
CLIP_AI_DAM/
├── DataSet/
│   ├── Images/                   # Place images here (JPG, PNG, WEBP, etc.)
│   └── Documents/                # Place documents here (PDF, DOCX, PPTX)
│
├── embeddings/
│   └── image_embeddings.pkl      # Auto-generated hybrid embedding cache
│
├── Output/
│   ├── FlowDiagram.png
│   ├── searchingImages.png
│   ├── searchingForDocuments.png
│   └── analyticsDashboard.png
│
├── src/
│   ├── pages/
│   │   └── analytics.py          # System & dataset analytics dashboard
│   ├── app.py                    # Streamlit application entry point
│   ├── document_search.py        # Document pipeline: extract, chunk, embed, FAISS, rerank
│   └── encode_images.py          # BLIP captioning + CLIP hybrid encoding
│
├── requirements.txt
├── environment.yml
└── README.md
```

> `DataSet/` and `image_embeddings.pkl` are excluded from the repository. Add your own files locally before running the app.

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Suhani6245/DigitalAssetManager.git
cd CLIP_AI_DAM
```

### 2. Option A — pip (virtual environment)

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### 2. Option B — Conda

```bash
conda env create -f environment.yml
conda activate ai-dam
```

> **macOS (Apple Silicon):** If `faiss-cpu` fails via pip:
> ```bash
> conda install -c conda-forge faiss-cpu
> ```

### 3. Add your data

```
DataSet/Images/       ← JPG, JPEG, PNG, BMP, WEBP, GIF
DataSet/Documents/    ← PDF, DOCX, PPTX
```

### 4. Generate image embeddings (run once)

```bash
python src/encode_images.py
```

This will:
- Generate BLIP captions for every image automatically
- Compute hybrid CLIP visual + caption embeddings
- Save everything to `embeddings/image_embeddings.pkl`
- Skip already-encoded images on future runs
- Unload BLIP from memory when done (~900 MB freed)

### 5. Run the application

```bash
streamlit run src/app.py
# or
python -m streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How It Works

### Image Search (Hybrid)

1. BLIP generates a natural language caption for each image offline.
2. CLIP encodes both the image visually and the caption as text.
3. The two vectors are averaged and L2-normalised into a single hybrid embedding.
4. At search time, the query is encoded with CLIP's text encoder.
5. A vectorized NumPy dot-product scores all images simultaneously.
6. Top-K results shown with similarity scores and captions in the UI.

### Document Search

1. Text extracted from PDF / DOCX / PPTX using format-specific parsers.
2. Cleaned and split into overlapping sentence-based chunks.
3. Chunks embedded with `BAAI/bge-base-en-v1.5` and indexed in FAISS.
4. Top candidates retrieved from FAISS, then reranked by CrossEncoder.
5. Best-matching sentence highlighted per result in the UI.

### Unified Search

Image captions are injected into the document FAISS index alongside document chunks, enabling a single query to surface both images and documents through one semantic pipeline.

### Flow Diagram

![Flow Diagram](Output/FlowDiagram.png)

---

## Model Storage

| Model | Size | When Loaded |
|---|---|---|
| CLIP ViT-B/32 | ~350 MB | Always (app start) |
| BAAI/bge-base-en-v1.5 | ~420 MB | Always (app start) |
| CrossEncoder MiniLM | ~85 MB | Always (app start) |
| BLIP captioning-base | ~900 MB | Encoding only — unloaded after |
| **Search-time total** | **~855 MB** | |
| **Encoding-time total** | **~1.75 GB** | |

Models are downloaded once and cached by HuggingFace at `~/.cache/huggingface/`.

---

## Tuning

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `BATCH_SIZE` | `encode_images.py` | 4 | BLIP batch size — increase if RAM allows |
| `SAVE_EVERY` | `encode_images.py` | 200 | Checkpoint interval during encoding |
| `chunk_size` | `document_search.py` | 5 sentences | Larger = more context per chunk |
| `overlap` | `document_search.py` | 2 sentences | Reduces boundary misses |
| `top_k * 10` | `search()` | ×10 candidates | Candidates fed to CrossEncoder reranker |
| Top-K slider | Streamlit sidebar | 5 | Final results shown in UI |

---

## Resetting the Image Cache

Delete `embeddings/image_embeddings.pkl` and re-run `encode_images.py`.
To add only new images without rebuilding from scratch, add files and re-run — existing images are skipped automatically.

---

## Dependencies at a Glance

```
streamlit             ← UI + analytics dashboard
torch / torchvision   ← Deep learning runtime
openai-clip           ← Image-text matching
transformers          ← BLIP image captioning
accelerate            ← Faster HuggingFace inference
Pillow                ← Image I/O
numpy                 ← Numerics
tqdm                  ← Encoding progress bar
sentence-transformers ← Bi-encoder + CrossEncoder
faiss-cpu             ← Vector index
pymupdf (fitz)        ← PDF parsing
python-docx           ← DOCX parsing
python-pptx           ← PPTX parsing
```

---

## Screenshots

![Searching Images](Output/searchingImages.png)

![Searching Documents](Output/searchingForDocuments.png)

![Analytics Dashboard](Output/analyticsDashboard.png)

---

## Datasets Used

| Asset Type | Source | Included in Repo |
|---|---|---|
| Images | Flickr8k dataset | No |
| Documents | Local personal/test files | No |

> Both datasets are excluded from the repository for size and privacy reasons. Users are expected to supply their own files.

---

## Privacy

All processing is local. No images, documents, queries, or embeddings are transmitted to any external service.

---

## Limitations

- BLIP encoding is slow on CPU for large image libraries. GPU recommended for 1000+ images.
- Scanned PDFs without embedded text are not supported (no OCR).
- Document index is rebuilt in memory at each app start. Startup time increases with library size.
- Optimized for English-language content.

---

## Future Improvements

- Persistent FAISS document index saved to disk to avoid rebuilding on restart.
- OCR support for scanned PDF documents.
- Tunable CLIP visual vs. caption embedding weight ratio.
- Metadata filtering (file type, date range, size) alongside semantic search.
- Multi-language query support.
- BLIP `fp16` half-precision for faster GPU encoding.
- Web-accessible deployment with user authentication.

---

## Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Salesforce BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS — Meta AI](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)