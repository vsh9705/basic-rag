# server/modules/policy_store.py
import os
from pathlib import Path
import shutil
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parents[1]
PERSIST_POLICY_DIR = str(BASE_DIR / "policy_store")
POLICY_UPLOAD_DIR = str(BASE_DIR / "uploaded_policies")

os.makedirs(POLICY_UPLOAD_DIR, exist_ok=True)
os.makedirs(PERSIST_POLICY_DIR, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L12-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def save_policy_files(uploaded_files) -> list[str]:
    """
    Save uploaded policy files to disk and return list of saved paths.
    """
    file_paths = []
    for file in uploaded_files:
        save_path = Path(POLICY_UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))
    return file_paths

def build_or_update_policy_store(uploaded_files):
    """
    Save uploaded policy PDFs, split into chunks, embed, and persist to policy_store.
    Appends to existing store if present.
    Returns the Chroma vectorstore instance.
    """
    file_paths = save_policy_files(uploaded_files)

    docs = []
    for p in file_paths:
        loader = PyPDFLoader(p)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_POLICY_DIR) and os.listdir(PERSIST_POLICY_DIR):
        # load and add
        store = Chroma(persist_directory=PERSIST_POLICY_DIR, embedding_function=embeddings)
        store.add_documents(texts)
        store.persist()
    else:
        # create new
        store = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_POLICY_DIR)
        store.persist()

    return store

def load_policy_store():
    """
    Load existing policy store. Raises FileNotFoundError if empty.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(PERSIST_POLICY_DIR) and os.listdir(PERSIST_POLICY_DIR):
        return Chroma(persist_directory=PERSIST_POLICY_DIR, embedding_function=embeddings)
    else:
        raise FileNotFoundError("Policy store is empty. Upload policy PDFs first.")

def clear_policy_store():
    """
    Remove the persisted policy store and policy upload folder (full reset).
    """
    if os.path.exists(PERSIST_POLICY_DIR):
        shutil.rmtree(PERSIST_POLICY_DIR)
    if os.path.exists(POLICY_UPLOAD_DIR):
        shutil.rmtree(POLICY_UPLOAD_DIR)
    # recreate empty directories
    os.makedirs(POLICY_UPLOAD_DIR, exist_ok=True)
    os.makedirs(PERSIST_POLICY_DIR, exist_ok=True)
    return True
