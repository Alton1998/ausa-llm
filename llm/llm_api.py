import logging
import os
import pickle

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from langchain_community.llms import LlamaCpp

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
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging_level,
)

app = FastAPI()

MONGO_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
DB_NAME = os.getenv("MONGODB_NAME")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("VECTOR_SEARCH_INDEX", "medical_info_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "thenlper/gte-large")
LLM_MODEL_TEMPERATURE = float(os.getenv("LLM_MODEL_TEMPERATURE", 0.2))
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "openbiollm-llama3-8b.Q5_K_M.gguf")
LLM_MODEL_MAX_TOKENS = int(os.getenv("LLM_MODEL_MAX_TOKENS", 2000))
LLM_MODEL_TOP_P = float(os.getenv("LLM_MODEL_TOP_P", 1))


f = open(os.getenv("EMBEDDING_MODEL_PATH","./embedder"),"rb")
embeddings = pickle.load(f)
f.close()


vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    MONGO_URI,
    DB_NAME + "." + COLLECTION_NAME,
    embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

rag_prompt = PromptTemplate.from_template(
    "You are an expert and experienced from the healthcare and biomedical domain with extensive medical "
    "knowledge and practical experience. Your name is Stephen, and you were developed by AUSA. You are"
    "willing to help answer the user's query with explanation. In your explanation, leverage your deep "
    "medical expertise such as relevant anatomical structures, physiological processes, diagnostic "
    "criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology "
    "while still aiming to make the explanation clear and accessible to a general audience. If you think "
    "its necessary use given context as well to answer the questions:{context}. If you don't know anything "
    "just say I don't know. "
    "Also end your answer with a caution message saying that your answers may not be fully accurate"
    "Medical Question: {question} Medical Answer:"
)


def format_docs(docs):
    return " ".join(doc.page_content for doc in docs)


llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
    temperature=LLM_MODEL_TEMPERATURE,
    max_tokens=LLM_MODEL_MAX_TOKENS,
    top_p=LLM_MODEL_TOP_P,
    n_ctx=2048,
)

chain = (
    RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
    | rag_prompt
    | llm
    | StrOutputParser()
)

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <h2>Your ID: <span id="ws-id"></span></h2>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var client_id = Date.now()
            document.querySelector("#ws-id").textContent = client_id;
            var ws = new WebSocket(`ws://localhost:8000/llm/${client_id}`);
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


class LLMConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


@app.get("/")
async def get():
    return HTMLResponse(html)


llm_connection_manager = LLMConnectionManager()


@app.websocket("/llm/{client_id}")
async def llm_websocket(websocket: WebSocket, client_id: int):
    logging.info("Came here")
    await llm_connection_manager.connect(websocket)
    logging.debug(f"Client:{client_id} Connected")
    try:
        while True:
            question = await websocket.receive_text()
            logging.debug(f"Client:{client_id} asked: {question}")
            docs = vector_search.similarity_search(question)
            logging.debug(f"Documents Found:{len(docs)}")
            response = chain.invoke({"question": question, "context": docs})
            await llm_connection_manager.send_personal_message(response, websocket)
    except WebSocketDisconnect:
        await llm_connection_manager.disconnect(websocket)
        logging.debug(f"Client {client_id} Disconnected")
