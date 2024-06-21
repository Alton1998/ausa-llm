import os
from typing import Any, Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from pymongo import MongoClient



class GetPatientVitalsWithID(BaseModel):
    user_id: str = Field(description="Patient ID")


class GetPatientVitalsWithName(BaseModel):
    user_name: str = Field(description="Patient Name")


class GetPatientVitalsWithUserNameTool(BaseTool):
    name:str = "GetPatientVitalsWithUserNameTool"
    __client:MongoClient = None
    __USER_COLLECTION = "users"
    __ENCOUNTER_COLLECTION = "encounters"

    def __initVectorDB(self):
        """Singleton pattern to initiate a Database"""
        if (not self.__client):
            self.__client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))
        return self.__client

    def _run(
        self,
        user_name: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        client = self.__client
        db = client[os.getenv("MONGODB_NAME")]
        user_collection = db[self.__USER_COLLECTION]
        encounter_collection = db[self.__ENCOUNTER_COLLECTION]
        # First Checking if there are any users with
        """Fetch User Vitals Synchronously"""

    def _arun(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:


