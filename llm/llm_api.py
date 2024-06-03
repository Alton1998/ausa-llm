import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
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

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    MONGO_URI,
    DB_NAME + "." + COLLECTION_NAME,
    embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

rag_prompt = PromptTemplate.from_template(
    "You are an assistant for medical question-answering tasks. Use the following "
    "pieces of retrieved context to answer the question. If you don't know the "
    "answer, just say that you don't know. Use three sentences maximum and keep "
    "the answer concise for upto 50 words. Question: {question} \nContext: {context}"
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
    temperature=LLM_MODEL_TEMPERATURE,
    max_tokens=LLM_MODEL_MAX_TOKENS,
    top_p=LLM_MODEL_TOP_P,
    callback_manager=callback_manager,
    verbose=True,
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
            response = chain.invoke({"context": docs, "question": question})
            await llm_connection_manager.send_personal_message(response, websocket)
    except WebSocketDisconnect:
        await llm_connection_manager.disconnect(websocket)
        logging.debug(f"Client {client_id} Disconnected")
