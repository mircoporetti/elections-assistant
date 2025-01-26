import logging
import os

from langchain_community.vectorstores import FAISS, VectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("uvicorn")

PROGRAMS_PATH = "./resources/manifests/"
vector_store: VectorStore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def faiss_folder_is_empty_except_lockfile():
    return os.path.exists("faiss") and any(file for file in os.listdir("faiss") if file not in "init.lock")


def init_vector_store():
    global vector_store
    if faiss_folder_is_empty_except_lockfile():
        logger.info("Loading existing Vector Store...")
        vector_store = FAISS.load_local("faiss", embeddings, allow_dangerous_deserialization=True)
    else:
        logger.info("Creating new Vector Store from parties' programs docs...")
        docs_chunks = chunk_manifests_pdfs()
        vector_store = build_from_documents(docs_chunks)


def get_store_as_retriever():
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def build_from_documents(docs_chunks):
    store = FAISS.from_documents(docs_chunks, embeddings)
    store.save_local("faiss")
    return store


def chunk_manifests_pdfs():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    loader = PyPDFDirectoryLoader(PROGRAMS_PATH)
    docs_to_be_chunked = loader.load()
    return splitter.split_documents(docs_to_be_chunked)


def similarity_search(query):
    most_similar_results = vector_store.similarity_search_with_score(query, 4)
    return [
        {
            "text": result[0],
            "score": float(result[1])
        }
        for result in most_similar_results
    ]
