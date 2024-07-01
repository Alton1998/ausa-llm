import os

import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv

# load_dotenv()

# client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))
# db = client[os.getenv("MONGODB_NAME")]
# name = "Grace Evans"
# users_count = db["users"].count_documents({"name": {"$regex": name}})
# users = db["users"].find_one({"name": {"$regex": name}})
# encounters = (
#     db["encounters"]
#     .find({"user_id": users.get("id")})
#     .sort("date(UTC)", pymongo.DESCENDING)
# )
#
# print(users)
# print(users_count)
# import re
#
# tool_pattern = r"[']tool[']:[']([a-z]|[A-Z])*[']"
# json_string = "{'tool':'GetPatientEncounterDetailsTool','arguments':['Mathew','10']}"
# argument_pattern = r"[']arguments[']:\[(\w|,|')*\]"



import re
pattern = r'[A-Z][a-z]*'
name_string = "GraceEvans"
match = re.finditer(pattern, name_string)
for mat in match:
    print(mat.group())


