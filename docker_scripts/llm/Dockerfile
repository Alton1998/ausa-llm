FROM python:3.12.3-slim
WORKDIR /code
COPY ./docker_scripts/llm/Pipfile Pipfile
COPY ./llm ./llm
RUN apt-get update && apt-get install build-essential -y
RUN pip install pipenv
RUN pipenv install --deploy
CMD ["pipenv","run","fastapi", "run", "/code/llm/llm_server.py", "--port", "8000"]
