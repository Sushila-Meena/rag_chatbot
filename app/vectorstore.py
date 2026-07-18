from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app.config import (
    COHERE_API_KEY,
    COHERE_EMBED_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    TOP_K,
)


_embeddings = CohereEmbeddings(
    model=COHERE_EMBED_MODEL,
    cohere_api_key=COHERE_API_KEY,
)

_pc    = Pinecone(api_key=PINECONE_API_KEY)
_index = _pc.Index(PINECONE_INDEX_NAME)

# vector store is set up lazily in build_index() so the app can start
# even before any documents are indexed
_vector_store: PineconeVectorStore | None = None


def build_index(chunks: list[Document]) -> None:
    """
    embeds chunks via cohere and upserts them into the pinecone namespace.
    wipes the namespace first so re-indexing on restart doesn't stack duplicates.
    """
    global _vector_store

    # clear existing vectors in this namespace to avoid duplicates
    try:
        _index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        print(f"[vectorstore] cleared namespace '{PINECONE_NAMESPACE}'")
    except Exception:
        pass  # namespace may not exist on first run

    _vector_store = PineconeVectorStore(
        embedding=_embeddings,
        index=_index,
        namespace=PINECONE_NAMESPACE,
    )

    ids = _vector_store.add_documents(documents=chunks)
    print(f"[vectorstore] upserted {len(ids)} vectors")


def retrieve(query: str, k: int = TOP_K) -> list[Document]:
    """
    embeds the query and returns the top-k most similar chunks.
    raises if the index has not been built yet.
    """
    if _vector_store is None:
        raise RuntimeError("vector store not initialised — call build_index() first")

    return _vector_store.similarity_search(query, k=k)
