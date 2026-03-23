import chromadb
from chromadb.config import Settings as ChromaSettings
from google import genai
from google.genai import types
from src.chunker import Chunk
from src.config import settings

COLLECTION_NAME = "rag_documents"
# Gemini embedding API batch limit
_EMBED_BATCH_SIZE = 100


class VectorStore:
    def __init__(self):
        self._genai = genai.Client(api_key=settings.gemini_api_key)
        self._chroma = chromadb.Client(ChromaSettings(anonymized_telemetry=False))
        self._collection = self._chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # Embedding helpers
    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts in batches."""
        embeddings = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i: i + _EMBED_BATCH_SIZE]
            result = self._genai.models.embed_content(
                model=settings.embedding_model,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            embeddings.extend([e.values for e in result.embeddings])
        return embeddings

    def _embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        result = self._genai.models.embed_content(
            model=settings.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return result.embeddings[0].values

    
    # Public interfaces
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and store a list of chunks with their metadata."""
        texts = [c.text for c in chunks]
        embeddings = self._embed_documents(texts)

        self._collection.add(
            ids=[f"{c.source_file}_p{c.page_number}_c{c.chunk_index}" for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"source_file": c.source_file, "page_number": c.page_number}
                for c in chunks
            ],
        )

    def query(self, question: str) -> list[dict]:
        """
        Return the top-k most relevant chunks for a question.
        Each result dict contains: text, source_file, page_number.
        """
        query_embedding = self._embed_query(question)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=settings.top_k_results,
            include=["documents", "metadatas"],
        )

        return [
            {"text": doc, "source_file": meta["source_file"], "page_number": meta["page_number"]}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    def reset(self) -> None:
        """Drop and recreate the collection (used when new documents are uploaded)."""
        self._chroma.delete_collection(COLLECTION_NAME)
        self._collection = self._chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def is_empty(self) -> bool:
        return self._collection.count() == 0
