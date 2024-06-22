import os

import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))
db = client[os.getenv("MONGODB_NAME")]
name = "Christine"
users_count = db["users"].count_documents({"name": {"$regex": name}})
users = db["users"].find_one({"name": {"$regex": name}})
encounters = (
    db["encounters"]
    .find({"user_id": users.get("id")})
    .sort("date(UTC)", pymongo.DESCENDING)
)
