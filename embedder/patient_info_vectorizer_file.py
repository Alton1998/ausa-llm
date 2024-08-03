import os
import pickle
from enum import Enum

import pymongo
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient

load_dotenv()
MONGO_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
CHUNK_SIZE = 300
CHUNK_OVERLAP = 0

client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))
db = client[os.getenv("MONGODB_NAME")]
MONGODB_COLLECTION = db[os.getenv("MONGODB_NAME")]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


class MongoDBCollections(Enum):
    USER_COLLECTION: str = "users"
    ENCOUNTERS: str = "encounters"


class MongoDBInstance:
    _client: MongoClient = None

    @classmethod
    def get_client(cls):
        if not cls._client:
            cls._client = MongoClient(MONGO_URI)
        return cls._client


client = MongoDBInstance.get_client()
db = client[os.getenv("MONGODB_NAME")]
user_collection = db[MongoDBCollections.USER_COLLECTION.value]
encounter_collection = db[MongoDBCollections.ENCOUNTERS.value]

encounters = encounter_collection.find().sort("date(UTC)", pymongo.DESCENDING)

encounter_summaries = []


def format_time(dt):
    """Formats a datetime object to the desired format."""
    ordinal_suffix = {1: "st", 2: "nd", 3: "rd"}.get(
        dt.day if dt.day < 20 else dt.day % 10, "th"
    )
    formatted_str = dt.strftime(f"%B {dt.day}{ordinal_suffix}, %Y %H:%M UTC")
    return formatted_str


for encounter in encounters:
    user = user_collection.find_one({"id": encounter["user_id"]})
    encounter_summary = f"""
    The patients name is {user["name"]}. The patients age is {user["age"]}. During this medical encounter which can 
    be identified with the encounter identification number {encounter["encounter_id"]} which took place on {format_time(encounter["date(UTC)"])}.
    The vitals observed for the patient are as follows :
    """
    for index, (key, value) in enumerate(encounter.get("vitals").items()):
        encounter_summary = (
            encounter_summary + str(index) + ")" + key + ":" + str(value) + "\n"
        )
    encounter_summary = (
        encounter_summary
        + f"""
    
    Where systolic_bp stands for "Systolic Blood Pressure", similarly diastolic_bp stands for "Diastolic Blood 
    Pressure" and SPO2 stands for "blood oxygen saturation levels in percentage" and HR stands for "heart rate". 
    
    For this Encounter the doctors comments where as follows:
    
    {encounter["comments"]}.
    
    Additionally the doctor also prescribed the following:
    
    {encounter["prescription"]}
    
    You Should summarise this information for the patient.
    
    """
    )
    len(encounter_summary)
    encounter_summaries.append(encounter_summary)

f = open(os.getenv("EMBEDDING_MODEL_PATH", "./embedder"), "rb")
embeddings = pickle.load(f)
f.close()


def embed_and_persist(
    strings,
    embedding_model=embeddings,
    mongo_uri=MONGO_URI,
    db_name=os.getenv("MONGODB_NAME"),
    collection_name="patient_encounter_embeddings",
):
    """Embeds a list of strings, stores the embeddings and original strings in MongoDB.

    Args:
      strings: A list of strings to be embedded.
      embedding_model: The embedding model to use. Defaults to OpenAIEmbeddings().
      mongo_uri: The MongoDB connection URI.
      db_name: The name of the MongoDB database.
      collection_name: The name of the MongoDB collection.
    """

    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    embeddings_local = embedding_model.embed_documents(strings)

    data = [
        {"text": text, "embedding": embedding}
        for text, embedding in zip(strings, embeddings_local)
    ]

    collection.insert_many(data)


embed_and_persist(encounter_summaries)
