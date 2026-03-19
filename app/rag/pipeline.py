from __future__ import annotations

import hashlib
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from app.rag.retriever import get_retriever, add_documents
from app.config import get_settings

settings = get_settings()

FINANCIAL_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional financial analyst with expertise in reading SEC filings, annual reports, and regulatory documents.
Use the following context extracted from financial documents to answer the question accurately.
If the answer is not clearly supported by the context, explicitly state that and provide what limited insight you can.
Always cite specific sections or figures when available.
Never fabricate financial data or statistics.

Context:
{context}""",
    ),
    ("human", "{input}"),
])


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )


def chunk_document(
    content: str,
    metadata: dict[str, Any],
) -> list[LCDocument]:
    splitter = get_text_splitter()
    chunks = splitter.create_documents([content], metadatas=[metadata])
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["content_hash"] = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:16]
    return chunks


def ingest_document(content: str, metadata: dict[str, Any]) -> list[str]:
    chunks = chunk_document(content, metadata)
    ids = add_documents(chunks)
    return ids


def get_rag_chain():
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,
        openai_api_key=settings.openai_api_key,
    )
    retriever = get_retriever()
    qa_chain = create_stuff_documents_chain(llm, FINANCIAL_RAG_PROMPT)
    return create_retrieval_chain(retriever, qa_chain)


def query_rag(question: str) -> dict[str, Any]:
    chain = get_rag_chain()
    result = chain.invoke({"input": question})
    sources = [
        {
            "content": doc.page_content[:200],
            "metadata": doc.metadata,
        }
        for doc in result.get("context", [])
    ]
    return {
        "answer": result["answer"],
        "sources": sources,
    }
