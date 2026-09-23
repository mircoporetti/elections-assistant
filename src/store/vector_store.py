from filelock import FileLock
import logging
import os

from langchain_community.vectorstores import FAISS, VectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chat.party import Party

logger = logging.getLogger("uvicorn")

PROGRAMS_PATH = "resources/manifests/"
FAISS_PATH = "faiss"
INDEX_FILE = os.path.join(FAISS_PATH, "index.faiss")
MODEL_MARKER = os.path.join(FAISS_PATH, "embedding_model.txt")
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

vector_store: VectorStore


class E5Embeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        return super().embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


embeddings = E5Embeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)


def index_exists():
    return os.path.exists(INDEX_FILE)


def index_was_built_with_current_model():
    try:
        with open(MODEL_MARKER) as marker:
            return marker.read().strip() == EMBEDDING_MODEL
    except OSError:
        return False


def init():
    global vector_store
    os.makedirs(FAISS_PATH, exist_ok=True)
    lock = FileLock(f"./{FAISS_PATH}/init.lock", timeout=120)
    with lock:
        logger.info("Initializing Vector Store...")
        if index_exists() and index_was_built_with_current_model():
            logger.info("Loading existing Vector Store...")
            vector_store = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            if index_exists():
                logger.info("Existing Vector Store was built with a different embedding model, rebuilding...")
            else:
                logger.info("Creating new Vector Store from parties' programs docs...")
            vector_store = build_from_documents(chunk_manifests_pdfs())
        logger.info("Vector Store has been initialized.")


def build_from_documents(docs_chunks):
    store = FAISS.from_documents(docs_chunks, embeddings)
    store.save_local(FAISS_PATH)
    with open(MODEL_MARKER, "w") as marker:
        marker.write(EMBEDDING_MODEL)
    return store


def chunk_manifests_pdfs():
    splitter = RecursiveCharacterTextSplitter(chunk_size=2200, chunk_overlap=200)
    loader = PyPDFDirectoryLoader(f"./{PROGRAMS_PATH}")
    docs_to_be_chunked = loader.load()
    return splitter.split_documents(docs_to_be_chunked)


def clean():
    os.makedirs(FAISS_PATH, exist_ok=True)
    lock = FileLock(f"./{FAISS_PATH}/init.lock", timeout=120)
    with lock:
        if not index_exists():
            logger.info("Nothing to clean...")
        else:
            logger.info("Cleaning up Vector Store...")
            for filename in os.listdir(FAISS_PATH):
                if filename == "init.lock":
                    continue
                file_path = os.path.join(FAISS_PATH, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        logger.info("Vector Store cleanup successful.")


def party_filter_for(party: Party):
    return {"source": f"{PROGRAMS_PATH}{party.name}.pdf"} if party else {}


def fetch_k_for(party_filter):
    if not party_filter:
        return 20
    return max(20, vector_store.index.ntotal)


def get_store_as_retriever_for(party: Party):
    party_filter = party_filter_for(party)
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4, "filter": party_filter, "fetch_k": fetch_k_for(party_filter)},
    )


def similarity_search_for(party: Party, query: str):
    party_filter = party_filter_for(party)
    most_similar_results = vector_store.similarity_search_with_score(
        query, k=4, filter=party_filter, fetch_k=fetch_k_for(party_filter)
    )
    return [
        {
            "text": result[0],
            "score": float(result[1])
        }
        for result in most_similar_results
    ]
