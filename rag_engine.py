from pathlib import Path
import hashlib
import re
import shutil


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
MODELS_DIR = BASE_DIR / "models"
LOCAL_EMBEDDING_MODEL_DIR = MODELS_DIR / "all-MiniLM-L6-v2"
COLLECTION_NAME = "local_knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_EMBEDDING_MODEL = None


def ensure_rag_folders():
    KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
    CHROMA_DB_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)


def _load_chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "ChromaDB is not installed. Install requirements.txt first: chromadb, sentence-transformers, pypdf."
        ) from exc

    return chromadb


def _load_pypdf():
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is not installed. Install requirements.txt first.") from exc

    return PdfReader


def get_embedding_model():
    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is not installed. Install requirements.txt first.") from exc

    try:
        # Embeddings turn text into numeric vectors. local_files_only keeps the app offline.
        if LOCAL_EMBEDDING_MODEL_DIR.exists():
            _EMBEDDING_MODEL = SentenceTransformer(str(LOCAL_EMBEDDING_MODEL_DIR), local_files_only=True)
        else:
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except TypeError:
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Embedding model all-MiniLM-L6-v2 was not found locally. "
            "Download/cache it once before running fully offline."
        ) from exc

    return _EMBEDDING_MODEL


def get_collection():
    ensure_rag_folders()
    chromadb = _load_chromadb()

    # PersistentClient stores vectors on disk in chroma_db/, so the index survives app restarts.
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def list_knowledge_files():
    ensure_rag_folders()
    files = []

    for path in KNOWLEDGE_BASE_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return sorted(files)


def read_text_file(path):
    # errors="replace" keeps old/odd encodings from crashing ingestion.
    return path.read_text(encoding="utf-8", errors="replace")


def read_pdf_file(path):
    PdfReader = _load_pypdf()
    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[Page {page_number}]\n{page_text}")

    return "\n\n".join(pages)


def read_document(path):
    extension = path.suffix.lower()

    if extension in {".txt", ".md"}:
        return read_text_file(path)

    if extension == ".pdf":
        return read_pdf_file(path)

    return ""


def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    # Chunking breaks long documents into smaller pieces so vector search can find precise passages.
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def make_chunk_id(source_name, chunk_index, chunk_text):
    raw_id = f"{source_name}:{chunk_index}:{chunk_text}".encode("utf-8", errors="replace")
    return hashlib.sha1(raw_id).hexdigest()


def ingest_documents():
    ensure_rag_folders()
    files = list_knowledge_files()

    if not files:
        return {
            "documents": 0,
            "chunks": 0,
            "message": f"No .txt, .md, or .pdf files found in {KNOWLEDGE_BASE_DIR}.",
        }

    collection = get_collection()
    model = get_embedding_model()

    total_chunks = 0
    indexed_documents = 0

    for path in files:
        relative_source = str(path.relative_to(KNOWLEDGE_BASE_DIR))
        text = read_document(path)
        chunks = split_text_into_chunks(text)

        # Re-ingesting a document replaces old chunks for that source.
        try:
            collection.delete(where={"source": relative_source})
        except Exception:
            pass

        if not chunks:
            continue

        # Embeddings are numeric representations of text used for semantic similarity search.
        embeddings = model.encode(
            chunks,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        ids = [
            make_chunk_id(relative_source, chunk_index, chunk_text)
            for chunk_index, chunk_text in enumerate(chunks)
        ]
        metadatas = [
            {"source": relative_source, "chunk_index": chunk_index}
            for chunk_index in range(len(chunks))
        ]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)
        indexed_documents += 1

    return {
        "documents": indexed_documents,
        "chunks": total_chunks,
        "message": f"Indexed {indexed_documents} documents into {total_chunks} chunks.",
    }


def search_knowledge(query, top_k=4):
    ensure_rag_folders()
    collection = get_collection()
    indexed_chunks = collection.count()

    if indexed_chunks == 0:
        return []

    model = get_embedding_model()
    query_embedding = model.encode(
        [query],
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0].tolist()

    # Vector search compares the query embedding to stored chunk embeddings and returns closest chunks.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, indexed_chunks),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matches = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        matches.append({
            "text": document,
            "source": metadata.get("source", "unknown"),
            "chunk_index": metadata.get("chunk_index", 0),
            "distance": distance,
        })

    return matches


def format_retrieved_context(matches):
    if not matches:
        return "No relevant knowledge-base context was retrieved."

    formatted_chunks = []
    for index, match in enumerate(matches, start=1):
        formatted_chunks.append(
            f"[Knowledge chunk {index} | source: {match['source']} | chunk: {match['chunk_index']}]\n"
            f"{match['text']}"
        )

    return "\n\n".join(formatted_chunks)


def clear_vector_db():
    # Clearing removes local Chroma persistence. Source documents in knowledge_base/ are not deleted.
    if CHROMA_DB_DIR.exists():
        shutil.rmtree(CHROMA_DB_DIR)

    CHROMA_DB_DIR.mkdir(exist_ok=True)
    (CHROMA_DB_DIR / ".gitkeep").touch(exist_ok=True)

    return {"message": "Cleared local ChromaDB vector database."}


def get_indexed_documents_count():
    collection = get_collection()
    chunk_count = collection.count()

    if chunk_count == 0:
        return {"documents": 0, "chunks": 0}

    data = collection.get(include=["metadatas"])
    sources = {
        metadata.get("source")
        for metadata in data.get("metadatas", [])
        if metadata and metadata.get("source")
    }

    return {"documents": len(sources), "chunks": chunk_count}
