from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_embedder = None
_index = None
_chunks = []
_retriever_type = None
_retriever_backend = None
_tfidf_vectorizer = None
_tfidf_matrix = None
REGULATIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "regulations.txt"


def _load_regulations(filepath: str) -> list[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    sections = [s.strip() for s in text.split('\n\n') if s.strip()]
    return sections


def _build_index(chunks: list[str], embedder: Any, faiss_module: Any):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss_module.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))
    return index


def _build_tfidf_index(chunks: list[str]):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix


def _try_build_semantic_retriever(chunks: list[str]):
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError("Semantic retriever dependencies are unavailable.") from exc

    embedder = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
    index = _build_index(chunks, embedder, faiss)
    return embedder, index


def reset_retriever_cache() -> None:
    global _embedder, _index, _chunks, _retriever_type, _retriever_backend, _tfidf_vectorizer, _tfidf_matrix
    _embedder = None
    _index = None
    _chunks = []
    _retriever_type = None
    _retriever_backend = None
    _tfidf_vectorizer = None
    _tfidf_matrix = None


def get_retriever():
    global _embedder, _index, _chunks, _retriever_type, _retriever_backend, _tfidf_vectorizer, _tfidf_matrix

    if _retriever_type is not None:
        return {
            'type': _retriever_type,
            'backend': _retriever_backend,
            'embedder': _embedder,
            'index': _index,
            'chunks': _chunks,
            'vectorizer': _tfidf_vectorizer,
            'matrix': _tfidf_matrix,
        }

    _chunks = _load_regulations(str(REGULATIONS_PATH))

    try:
        _embedder, _index = _try_build_semantic_retriever(_chunks)
        _retriever_type = 'semantic'
        _retriever_backend = 'faiss_sentence_transformers'
    except Exception:
        _tfidf_vectorizer, _tfidf_matrix = _build_tfidf_index(_chunks)
        _retriever_type = 'lexical'
        _retriever_backend = 'tfidf'

    return {
        'type': _retriever_type,
        'backend': _retriever_backend,
        'embedder': _embedder,
        'index': _index,
        'chunks': _chunks,
        'vectorizer': _tfidf_vectorizer,
        'matrix': _tfidf_matrix,
    }


def retrieve(query: str, k: int = 3) -> str:
    retriever = get_retriever()
    chunks = retriever['chunks']

    if retriever['type'] == 'semantic':
        embedder = retriever['embedder']
        index = retriever['index']
        query_vec = embedder.encode([query], show_progress_bar=False).astype('float32')
        _, indices = index.search(query_vec, k)
        results = [chunks[i] for i in indices[0] if i < len(chunks)]
    else:
        vectorizer = retriever['vectorizer']
        matrix = retriever['matrix']
        query_vec = vectorizer.transform([query])
        scores = (matrix @ query_vec.T).toarray().ravel()
        indices = scores.argsort()[::-1][:k]
        results = [chunks[i] for i in indices if i < len(chunks) and scores[i] > 0]

        if not results:
            results = chunks[:k]

    return '\n\n---\n\n'.join(results)
