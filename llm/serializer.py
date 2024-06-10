import os
import pickle

from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "thenlper/gte-large")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

with open("embedder","ab") as f:
    pickle.dump(embeddings,f)


f = open("embedder","rb")
embeddings = pickle.load(f)
f.close()