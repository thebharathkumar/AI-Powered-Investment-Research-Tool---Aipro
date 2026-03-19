from __future__ import annotations

import hashlib
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from app.rag.retriever import get_retriever, add_documents
from app.config import get_settings

settings = get_settings()

FINANCIAL_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a professional financial analyst with expertise in reading SEC filings, annual reports, and regulatory documents.
Use the following context extracted from financial documents to answer the question accurately.
If the answer is not clearly supported by the context, explicitly state that and provide what limited insight you can.
Always cite specific sections or figures when available.
Never fabricate financial data or statistics.

Context:
{context}

Question: {question}

Structured Analysis:""",
)


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


def get_rag_chain() -> RetrievalQA:
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,
        openai_api_key=settings.openai_api_key,
    )
    retriever = get_retriever()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": FINANCIAL_RAG_PROMPT},
    )
    return chain


def query_rag(question: str) -> dict[str, Any]:
    chain = get_rag_chain()
    result = chain.invoke({"query": question})
    sources = [
        {
            "content": doc.page_content[:200],
            "metadata": doc.metadata,
        }
        for doc in result.get("source_documents", [])
    ]
    return {
        "answer": result["result"],
        "sources": sources,
    }
