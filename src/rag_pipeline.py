from google import genai

from src.config import settings
from src.vector_store import VectorStore

_PROMPT_TEMPLATE = """\
You are a helpful assistant that answers questions strictly based on the provided document excerpts.

Context excerpts:
{context}

Question: {question}

Instructions:
- Answer using only the information in the context above.
- If the answer cannot be found in the context, respond with: \
"I could not find an answer in the provided documents."
- Be concise and factual.
- Do not include citation markers inside the answer text; citations are shown separately.

Answer:\
"""


class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self._genai = genai.Client(api_key=settings.gemini_api_key)
        self._vector_store = vector_store

    def answer(self, question: str) -> dict:
        """
        Retrieve relevant chunks, generate an answer, and return citations.

        Returns:
            {
                "answer": str,
                "citations": [{"source_file": str, "page_number": int}, ...]
            }
        """
        retrieved_chunks = self._vector_store.query(question)

        # Build numbered context block for the prompt
        context = "\n\n".join(
            f"[{i + 1}] {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)
        )

        prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
        response = self._genai.models.generate_content(
            model=settings.llm_model,
            contents=prompt,
        )

        # Deduplicate citations while preserving order
        seen: set[tuple] = set()
        citations = []
        for chunk in retrieved_chunks:
            key = (chunk["source_file"], chunk["page_number"])
            if key not in seen:
                seen.add(key)
                citations.append({
                    "source_file": chunk["source_file"],
                    "page_number": chunk["page_number"],
                })

        return {"answer": response.text, "citations": citations}
