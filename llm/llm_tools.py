import os
from typing import Any, Optional, Type

import pymongo
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pymongo import MongoClient


class GetPatientVitalsWithID(BaseModel):
    user_id: str = Field(description="Patient ID")


class GetPatientVitalsWithName(BaseModel):
    user_name: str = Field(description="Patient Name")
    num_encounters: int = Field(description="Number of Encounters")


class GetPatientVitalsWithUserNameTool(BaseTool):
    name: str = "GetPatientVitalsWithUserNameTool"
    description: str = "Provided with the patient name and the number of encounters"
    args_schema: Type[BaseModel] = GetPatientVitalsWithName
    return_direct: bool = True
    __client: MongoClient = None
    __USER_COLLECTION = "users"
    __ENCOUNTER_COLLECTION = "encounters"

    def __init_vector_db(self):
        """Singleton pattern to initiate a Database"""
        if not self.__client:
            self.__client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))
        return self.__client

    def _run(
        self,
        user_name: str,
        num_encounters: int = 1,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Fetch User Vitals Synchronously"""
        client = self.__init_vector_db()
        db = client[os.getenv("MONGODB_NAME")]
        user_collection = db[self.__USER_COLLECTION]
        encounter_collection = db[self.__ENCOUNTER_COLLECTION]
        # First Checking if there are any users with
        user_query = {"name": {"$regex": user_name}}
        users_count = user_collection.count_documents(user_query)
        if users_count < 1:
            return "No user with such name exists as a result"
        user = user_collection.find_one(user_query)
        encounter_query = {"user_id": user.get("id")}
        encounters = encounter_collection.find(encounter_query).sort(
            "date(UTC)", pymongo.DESCENDING
        )
        response = ""

        for encounter in encounters:
            response = response + "____________________________________\n"
            response = response + "Encounter ID:" + encounter.get("id") + "\n"
            response = (
                response
                + "Date:"
                + encounter.get("date(UTC)").strftime("%d/%m/%Y, %H:%M:%S")
                + "\n"
            )
            response = response + "Vitals:\n"
            for index, (key, value) in enumerate(encounter.get("vitals").items()):
                response = response + str(index) + ")" + key + ":" + value + "\n"
            response = response + "Comments:" + encounter.get("comments") + "\n"
            response = response + "Prescription:" + encounter.get("prescription") + "\n"
        return response

    def _arun(
        self,
        user_name: str,
        num_encounters: int = 1,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Fetch Patient Vitals Asynchronously"""
        return self._run(
            user_name,
            num_encounters,
            run_manager.get_sync()
        )
