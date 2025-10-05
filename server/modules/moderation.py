import tempfile
from pathlib import Path
from typing import List, Dict
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from modules.llm import get_retrieval_qa_chain
from logger import logger

EMBEDDING_MODEL = "all-MiniLM-L12-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def load_current_pdf_to_chunks(uploaded_file) -> List[dict]:
    """
    Save the uploaded PDF temporarily, load it, split into chunks, and return a list of dicts.
    Each dict contains:
        - page_content: text content of chunk
        - metadata: chunk metadata including chunk_id
    """
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / uploaded_file.filename
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.file.read())

    loader = PyPDFLoader(str(temp_path))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(docs)

    chunk_dicts = []
    for i, c in enumerate(chunks):
        md = c.metadata.copy() if hasattr(c, "metadata") else {}
        md["chunk_id"] = f"{uploaded_file.filename}::chunk_{i}"
        chunk_dicts.append({"page_content": c.page_content, "metadata": md})

    return chunk_dicts


def moderate_file_against_policy(policy_store: Chroma, uploaded_file, k: int = 3) -> Dict:
    """
    For each chunk of the uploaded file:
    - Retrieve top-k policy snippets from the policy store
    - Use LLM (RetrievalQA) with custom moderation prompt to judge
    - Parse 'VIOLATION', 'REVIEW', or 'OK' verdict
    """
    chunks = load_current_pdf_to_chunks(uploaded_file)
    logger.debug(f"Split current file into {len(chunks)} chunks")

    # Initialize the RetrievalQA chain with custom moderation prompt
    chain = get_retrieval_qa_chain(policy_store, k=k, chain_type="stuff")

    violations = []
    allowed_count = 0
    review_count = 0

    for idx, chunk in enumerate(chunks):
        query_text = chunk["page_content"].strip()
        if not query_text:
            continue

        logger.debug(f"Moderating chunk {idx}: len={len(query_text)}")

        try:
            # Send chunk to LLM via RetrievalQA chain
            result = chain({"query": query_text})
            answer = result.get("result", "").strip()
            source_docs = result.get("source_documents", [])

            # Parse response based on updated prompt format
            answer_lower = answer.lower()
            if answer_lower.startswith("violation"):
                violations.append({
                    "chunk_id": chunk["metadata"].get("chunk_id"),
                    "chunk_text": query_text[:800],
                    "verdict": "violation",
                    "explanation": answer,
                    "sources": [d.metadata.get("source", "") for d in source_docs]
                })
            elif answer_lower.startswith("review"):
                review_count += 1
                violations.append({
                    "chunk_id": chunk["metadata"].get("chunk_id"),
                    "chunk_text": query_text[:800],
                    "verdict": "review",
                    "explanation": answer,
                    "sources": [d.metadata.get("source", "") for d in source_docs]
                })
            elif answer_lower.startswith("ok"):
                allowed_count += 1
            else:
                violations.append({
                    "chunk_id": chunk["metadata"].get("chunk_id"),
                    "chunk_text": query_text[:800],
                    "verdict": "unclear",
                    "explanation": answer,
                    "sources": []
                })

        except Exception as e:
            logger.exception("Error when running moderation chain on chunk")
            violations.append({
                "chunk_id": chunk["metadata"].get("chunk_id"),
                "chunk_text": query_text[:800],
                "verdict": "error",
                "explanation": str(e),
                "sources": []
            })

    total_chunks = len(chunks)
    verdict = "clean" if not violations else "violation_found"

    return {
        "verdict": verdict,
        "total_chunks": total_chunks,
        "allowed_chunks": allowed_count,
        "review_chunks": review_count,
        "violations": violations
    }
