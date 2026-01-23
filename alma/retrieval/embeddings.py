"""
ALMA Embedding Providers.

Supports local (sentence-transformers) and Azure OpenAI embeddings.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class LocalEmbedder(EmbeddingProvider):
    """
    Local embeddings using sentence-transformers.

    Default model: all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embedder.

        Args:
            model_name: Sentence-transformers model name
        """
        self.model_name = model_name
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded, dimension: {self._dimension}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )

    def encode(self, text: str) -> List[float]:
        """Generate embedding for text."""
        self._load_model()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self._load_model()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        if self._dimension is None:
            self._load_model()
        return self._dimension or 384  # Default for all-MiniLM-L6-v2


class AzureEmbedder(EmbeddingProvider):
    """
    Azure OpenAI embeddings.

    Uses text-embedding-3-small by default (1536 dimensions).
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment: str = "text-embedding-3-small",
        api_version: str = "2024-02-01",
    ):
        """
        Initialize Azure OpenAI embedder.

        Args:
            endpoint: Azure OpenAI endpoint (or use AZURE_OPENAI_ENDPOINT env var)
            api_key: Azure OpenAI API key (or use AZURE_OPENAI_KEY env var)
            deployment: Deployment name for embedding model
            api_version: API version
        """
        import os

        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        self.deployment = deployment
        self.api_version = api_version
        self._client = None
        self._dimension = 1536  # Default for text-embedding-3-small

        if not self.endpoint:
            raise ValueError(
                "Azure OpenAI endpoint required. Set AZURE_OPENAI_ENDPOINT env var "
                "or pass endpoint parameter."
            )

    def _get_client(self):
        """Get or create Azure OpenAI client."""
        if self._client is None:
            try:
                from openai import AzureOpenAI

                self._client = AzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            except ImportError:
                raise ImportError(
                    "openai is required for Azure embeddings. "
                    "Install with: pip install openai"
                )
        return self._client

    def encode(self, text: str) -> List[float]:
        """Generate embedding for text."""
        client = self._get_client()
        response = client.embeddings.create(
            input=text,
            model=self.deployment,
        )
        return response.data[0].embedding

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        client = self._get_client()
        response = client.embeddings.create(
            input=texts,
            model=self.deployment,
        )
        # Sort by index to ensure order matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


class MockEmbedder(EmbeddingProvider):
    """
    Mock embedder for testing.

    Generates deterministic fake embeddings based on text hash.
    """

    def __init__(self, dimension: int = 384):
        """Initialize mock embedder."""
        self._dimension = dimension

    def encode(self, text: str) -> List[float]:
        """Generate fake embedding based on text hash."""
        import hashlib

        # Create deterministic embedding from text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Use first N bytes to create float values
        embedding = []
        for i in range(self._dimension):
            byte_val = hash_bytes[i % len(hash_bytes)]
            # Normalize to [-1, 1] range
            embedding.append((byte_val / 127.5) - 1.0)
        return embedding

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate fake embeddings for multiple texts."""
        return [self.encode(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension
