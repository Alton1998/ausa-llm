import pickle
import re
from operator import itemgetter
from typing import List, Tuple
import json

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain import hub
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain.tools.render import render_text_description
from langchain.agents import create_structured_chat_agent, AgentExecutor

import logging
import os

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from llm_tools import GetPatientVitalsWithUserNameTool

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

MONGO_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
DB_NAME = os.getenv("MONGODB_NAME")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("VECTOR_SEARCH_INDEX", "medical_info_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "thenlper/gte-large")
LLM_MODEL_TEMPERATURE = float(os.getenv("LLM_MODEL_TEMPERATURE", 1.0))
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "openbiollm-llama3-8b.Q5_K_M.gguf")
LLM_MODEL_MAX_TOKENS = int(os.getenv("LLM_MODEL_MAX_TOKENS", 2000))
LLM_MODEL_TOP_P = float(os.getenv("LLM_MODEL_TOP_P", 1))

logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging_level,
)

f = open(os.getenv("EMBEDDING_MODEL_PATH", "./embedder"), "rb")
embeddings = pickle.load(f)
f.close()

tools = [GetPatientVitalsWithUserNameTool()]

rendered_tools_list = render_text_description(tools)

_TEMPLATE = """You are an expert and experienced from the healthcare and biomedical domain with extensive medical.
knowledge and practical experience. Your name is Stephen, and you were developed by AUSA. You are willing to help
answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant
anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical
concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a
general audience. If you think its necessary use given context as well to answer the questions. If you don't know
anything just say I don't know. Here are some details you might need:
Context:
{context}
User Question:
{question}
"""
PROMPT = PromptTemplate.from_template(_TEMPLATE)

vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
    MONGO_URI,
    DB_NAME + "." + COLLECTION_NAME,
    embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

retriever = vectorstore.as_retriever()

llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
    temperature=LLM_MODEL_TEMPERATURE,
    max_tokens=LLM_MODEL_MAX_TOKENS,
    top_p=LLM_MODEL_TOP_P,
    n_ctx=2048,
)


# User input
class LLMInput(BaseModel):
    """
    User Input data model
    """

    question: str


def format_docs(docs):
    return " ".join(doc.page_content for doc in docs)


conversational_qa_chain = (
    RunnableMap(
        question=itemgetter("question"),
        context=itemgetter("question") | retriever | format_docs,
    )
    | PROMPT
    | llm
    | StrOutputParser()
)


chain = conversational_qa_chain.with_types(input_type=LLMInput)

AGENT_PROMPT_TEMPLATE = """
Using the users query Determine the tool that needs to be used from the
given list of tools and their descriptions:

{tools}

Users Query:
{query}
"""

AGENT_PROMPT = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)

tool_determination_chain = AGENT_PROMPT | llm | StrOutputParser()

tool_pattern = r"[']tool[']:[']([a-z]|[A-Z])*[']"
argument_pattern = r"[']arguments[']:\[(\w|,|')*\]"


def extract_action(model_output: str):
    model_output = model_output.replace(" ", "")
    model_output = model_output.replace("{", "")
    model_output = model_output.replace("}", "")
    tool_match = re.search(tool_pattern, model_output)
    argument_match = re.search(argument_pattern, model_output)
    response = dict()
    response["tool"] = None
    if tool_match:
        response["tool"] = tool_match.group(0).split(":")[1]
    if argument_match:
        response["arguements"] = argument_match.group(0).split(":")[1]
    return response


def take_action(action_body):
    tool = action_body["tool"]
    patient_history = ""
    if tool is None:
        patient_history = "No patient Information available"
    if "GetPatientVitalsWithUserNameTool" in tool:
        action_body["arguements"] = action_body["arguements"].replace('"', "")
        action_body["arguements"] = action_body["arguements"].replace("[", "")
        action_body["arguements"] = action_body["arguements"].replace("]", "")
        name, encounter = action_body["arguements"].split(",")
        encounter = encounter.replace("'", "")
        patient_history = GetPatientVitalsWithUserNameTool().invoke(
            {"user_name": name, "num_encounters": int(encounter)}
        )
    return {"medical_information": patient_history}


SUMMARIZE_MEDICAL_DOCS_PROMPT_TEMPLATE = """
You are an expert and experienced from the healthcare and biomedical domain with extensive medical
knowledge and practical experience. Your name is Stephen, and you were developed by AUSA. In the following you are going to be given the medical history for a patient.
To the best of your knowledge summarize the information provided, highlighting any problems with the vitals and any corrective action that you might think is necessary.
Medical Information:
{medical_information}
"""

SUMMARIZE_MEDICAL_DOCS_PROMPT = PromptTemplate.from_template(
    SUMMARIZE_MEDICAL_DOCS_PROMPT_TEMPLATE
)


fetch_user_details_chain = (
    RunnableMap(
        query=itemgetter("question"),
        tools=lambda x: rendered_tools_list,
    )
    | tool_determination_chain
    | extract_action
    | take_action
    | SUMMARIZE_MEDICAL_DOCS_PROMPT
    | llm
    | StrOutputParser()
)

app = FastAPI(
    title="Ausa LLM",
    version="1.0",
    description="Ausa AI Co-pilot Service",
)
# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, chain, enable_feedback_endpoint=True, path="/general_info")
add_routes(
    app,
    fetch_user_details_chain,
    enable_feedback_endpoint=True,
    path="/summarize_user_reports",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
