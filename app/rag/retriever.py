from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LCDocument
from app.rag.embeddings import get_embeddings
from app.config import get_settings

settings = get_settings()


def get_vector_store() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def get_retriever(search_kwargs: dict | None = None):
    vector_store = get_vector_store()
    kwargs = search_kwargs or {"k": 6, "fetch_k": 20, "score_threshold": 0.5}
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=kwargs,
    )


def add_documents(documents: list[LCDocument]) -> list[str]:
    vector_store = get_vector_store()
    ids = vector_store.add_documents(documents)
    return ids


def similarity_search_with_score(query: str, k: int = 6) -> list[tuple[LCDocument, float]]:
    vector_store = get_vector_store()
    return vector_store.similarity_search_with_relevance_scores(query, k=k)
