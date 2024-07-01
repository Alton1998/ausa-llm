import logging
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
from enum import Enum


class GetPatientVitalsWithID(BaseModel):
    user_id: str = Field(description="Patient ID")


class GetPatientVitalsWithName(BaseModel):
    user_name: str = Field(description="Patient Name")
    num_encounters: int = Field(description="Number of Encounters")


class MongoDBCollections(Enum):
    USER_COLLECTION: str = "users"
    ENCOUNTERS: str = "encounters"


class MongoDBInstance:
    _client: MongoClient = None

    @classmethod
    def get_client(cls):
        if not cls._client:
            cls._client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))
        return cls._client


class GetPatientVitalsWithUserNameTool(BaseTool):
    name: str = "GetPatientVitalsWithUserNameTool"
    description: str = """Provided with the patient name and the number of encounters this tool will help fetch encounter details and vitals from the database.
    Example 1 :
    User Query: For a patient with name Grace Evans give me the last 5 encounters.
    Response:
    {
        'tool': 'GetPatientVitalsWithUserNameTool',
        'arguments':['Grace Evans','5']
    }

    Example 2:
    User Query: Give me the last 2 encounter details of Lata.
    Response:
    {
        'tool': 'GetPatientVitalsWithUserNameTool',
        'arguments':['Lata','2']
    }

    Example 3:
    User Query: Get me the last 4 encounter details of Alton Dsouza.
    Response:
    {
        'tool': 'GetPatientVitalsWithUserNameTool',
        'arguments':['Alton Dsouza','4']
    }

    Example 4:
    User Query: Get me the last 10 encounter details for Mathew.
    Response:
    {
        'tool': 'GetPatientVitalsWithUserNameTool',
        'arguments':['Mathew','10']
    }

    Example 5:
    User Query: Get me the last 15 encounters for Ayushi.
    Response:
    {
        'tool': 'GetPatientVitalsWithUserNameTool',
        'arguments':['Ayushi','15']
    }

    If the User's Query are anything like the given examples then GetPatientVitalsWithUserNameTool will be used. If they are not like the above examples do not use GetPatientVitalsWithUserNameTool.


    Other queries not matching the above examples do not need to use this tool.

    Example 1:
    I want a cherry.
    Response:
    None

    Example 2:
    What is heart disease?
    Response:
    None

    Example 3:
    Response:
    None

    What is AIDS?
    Response:
    None

    Example 4:
    Tell me About Cancer.
    Response:
    None

    All these are examples of when the GetPatientVitalsWithUserNameTool is not supposed to be used. Your Responses should follow the same as given in the examples.
    """
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
        print(user_name)
        client = MongoDBInstance.get_client()
        db = client[os.getenv("MONGODB_NAME")]
        user_collection = db[MongoDBCollections.USER_COLLECTION.value]
        encounter_collection = db[MongoDBCollections.ENCOUNTERS.value]
        # First Checking if there are any users with
        user_query = {"name": {"$regex": user_name}}
        users_count = user_collection.count_documents(user_query)
        if users_count < 1:
            return "No user with such name exists as a result"
        user = user_collection.find_one(user_query)
        print(user)
        encounter_query = {"user_id": user.get("id")}
        encounters = encounter_collection.find(encounter_query).sort(
            "date(UTC)", pymongo.DESCENDING
        )
        response = ""

        for encounter in encounters:
            response = response + "____________________________________\n"
            response = response + "Encounter ID:" + encounter.get("encounter_id") + "\n"
            response = (
                response
                + "Date:"
                + encounter.get("date(UTC)").strftime("%d/%m/%Y, %H:%M:%S")
                + "\n"
            )
            response = response + "Vitals:\n"
            for index, (key, value) in enumerate(encounter.get("vitals").items()):
                response = response + str(index) + ")" + key + ":" + str(value) + "\n"
            response = response + "Comments:" + encounter.get("comments") + "\n"
            response = response + "Prescription:" + encounter.get("prescription") + "\n"
            print(response)
        return response

    def _arun(
        self,
        user_name: str,
        num_encounters: int = 1,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Fetch Patient Vitals Asynchronously"""
        return self._run(user_name, num_encounters, run_manager.get_sync())
