FROM nvidia/cuda:12.5.0-devel-ubuntu22.04
WORKDIR /tmp
RUN apt-get update
RUN apt-get install -y build-essential git zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev curl
RUN wget https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz
RUN tar -xf Python-3.12.3.tgz
RUN cd Python-3.12.3/ && ./configure --enable-optimizations && make -j $(nproc) && make altinstall
WORKDIR /code
COPY ./docker_scripts/llm_gpu/Pipfile Pipfile
COPY ./llm ./llm
RUN pip3.12 install pipenv
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
RUN pipenv install --deploy
EXPOSE 8000
WORKDIR /code/llm
CMD ["pipenv","run","fastapi", "run", "llm_server.py", "--port", "8000"]
# CMD tail -f /dev/null
