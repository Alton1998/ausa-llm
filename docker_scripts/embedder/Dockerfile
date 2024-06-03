FROM python:3.12.3-slim
WORKDIR /code
COPY ./docker_scripts/embedder/Pipfile Pipfile
COPY ./embedder ./embedder
RUN pip install pipenv
RUN pipenv install --deploy
CMD ["pipenv","run","python","/code/embedder/vectorizer.py"]
