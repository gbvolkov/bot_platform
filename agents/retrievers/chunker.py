# chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain.docstore.document import Document

def chunk_text(text: str, chunk_size: int = 2048, chunk_overlap: int = 0) -> List[str]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    For each document, split its text into chunks (if needed) and return a new list of Documents.
    """
    chunked_docs = []
    for doc in docs:
        chunks = chunk_text(doc.page_content)
        for chunk in chunks:
            chunked_docs.append(
                Document(page_content=chunk, metadata=doc.metadata)
            )
    return chunked_docs
