"""Local sentence embeddings (optional). Falls back to lexical clustering when unavailable.

The model is downloaded once into the Hugging Face cache (~/.cache/huggingface) and loaded
from disk afterwards - no network call on later runs. It is also loaded lazily, only when a
phase actually needs embeddings (cluster / mapping), so resumed runs and --only runs skip it.
"""
from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer
        try:
            # Cached copy -> no Hub request at all.
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            log.info("Downloading embedding model %s (one-time)...", model_name)
            self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)


class LazyEmbedder:
    """Loads the model on first use. bool(lazy) is False if embeddings are unavailable."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._impl: Embedder | None = None
        self._failed = False

    def _load(self) -> Embedder | None:
        if self._impl is None and not self._failed:
            try:
                self._impl = Embedder(self.model_name)
            except ImportError:
                self._failed = True
                log.warning("sentence-transformers not installed -> lexical clustering only "
                            "(pip install 'kwresearch[embeddings]')")
            except Exception as e:  # download failure etc.
                self._failed = True
                log.warning("Embedding model unavailable (%s) -> lexical clustering only", e)
        return self._impl

    def __bool__(self) -> bool:
        return self._load() is not None

    def encode(self, texts: list[str]) -> np.ndarray:
        impl = self._load()
        if impl is None:
            raise RuntimeError("embeddings unavailable")
        return impl.encode(texts)


def get_embedder(model_name: str = DEFAULT_MODEL) -> LazyEmbedder:
    return LazyEmbedder(model_name)
