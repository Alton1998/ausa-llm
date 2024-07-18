ARG CUDA_IMAGE="12.4.1-cudnn-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}
WORKDIR /tmp
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git  build-essential procps file curl software-properties-common \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
ENV PATH="$PATH:/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin"
RUN brew install pyenv
RUN brew install gcc
WORKDIR /code
COPY ./docker_scripts/llm_gpu/Pipfile Pipfile
COPY ./llm ./llm
RUN pip3 install pipenv
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1
ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN pipenv install --deploy
EXPOSE 8000
WORKDIR /code/llm
CMD ["pipenv","run","fastapi", "run", "llm_server.py", "--port", "8000"]
# CMD tail -f /dev/null
