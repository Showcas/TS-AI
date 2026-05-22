FROM python:3.11-slim

WORKDIR /workspace

# Create a non-root user
ARG USERNAME=user
ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID $USERNAME \
    && useradd -m -u $UID -g $GID -s /bin/bash $USERNAME

RUN apt update -y
RUN apt upgrade -y
RUN apt install -y build-essential
RUN apt install -y libagg-dev
RUN apt install -y libcairo2
RUN apt install -y libpotrace-dev
RUN apt install -y pkg-config

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install "cairosvg>=2.8.2"
RUN pip install "matplotlib>=3.10.3"
RUN pip install "pillow>=11.3.0"
RUN pip install "pypotrace>=0.3"
RUN pip install "scipy>=1.16.0"
RUN pip install "svgelements>=1.9.6"
RUN pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
RUN pip install "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cpu
RUN pip install "tqdm==4.67.1" 
RUN pip install "numpy==2.3.4"

USER $USERNAME


ENTRYPOINT [ "/bin/bash" ]