from langchain_openai import OpenAIEmbeddings
from app.config import get_settings

settings = get_settings()


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        model="text-embedding-3-small",
        chunk_size=500,
    )
