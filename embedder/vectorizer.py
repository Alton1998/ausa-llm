import logging
import os
import pickle

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from more_itertools import chunked
from pymongo import MongoClient

load_dotenv()
logging_level = os.getenv("LOG_LEVEL", "INFO")
if logging_level == "INFO":
    logging_level = logging.INFO
elif logging_level == "WARN":
    logging_level = logging.WARNING
elif logging_level == "ERROR":
    logging_level = logging.ERROR
elif logging_level == "CRITICAL":
    logging_level = logging.CRITICAL
else:
    logging_level = logging.DEBUG

logging.basicConfig(
    filename="vectorizer.log",
    filemode="w",
    format="%(levelname)s:%(message)s",
    level=logging_level,
)

client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))
db = client[os.getenv("MONGODB_NAME")]
MONGODB_COLLECTION = db[os.getenv("MONGODB_COLLECTION_NAME")]
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("VECTOR_SEARCH_INDEX", "medical_info_index")

data = []
DIR = os.getenv("KNOWLEDGE_DIR", "./medical_training_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 20))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "thenlper/gte-large")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 10))

logging.info("Environment Variables:")
logging.info(f"ATLAS_VECTOR_SEARCH_INDEX_NAME:{ATLAS_VECTOR_SEARCH_INDEX_NAME}")
logging.info(f"DIR:{DIR}")
logging.info(f"CHUNK_SIZE:{CHUNK_SIZE}")
logging.info(f"CHUNK_OVERLAP:{CHUNK_OVERLAP}")
logging.info(f"EMBEDDING_MODEL:{EMBEDDING_MODEL}")

for current_path, folders, files in os.walk(DIR):
    files = filter(lambda x: x.endswith(".pdf"), files)
    for file in files:
        data.extend(PyPDFLoader(os.path.join(current_path, file)).load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
f = open(os.getenv("EMBEDDING_MODEL_PATH","./embedder"),"rb")
embeddings = pickle.load(f)
f.close()

for batch_no, batch in enumerate(chunked(data, BATCH_SIZE)):
    logging.info(f"Processing Batch Number:{batch_no}")
    logging.debug(f"Batch contains the following files:{batch}")
    docs = text_splitter.split_documents(batch)
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=embeddings,
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
