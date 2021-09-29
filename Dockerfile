FROM continuumio/miniconda3

WORKDIR /app

RUN apt-get update -y
# RUN apt-get install gcc g++ cmake vim -y
RUN apt-get install -y build-essential

RUN conda create -n pomdp python=3.8 -y

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "pomdp", "/bin/bash", "-c"]

RUN pip install Cython
RUN git clone https://github.com/h2r/pomdp-py.git
RUN cd pomdp-py/ && pip install -e.
WORKDIR /app/pomdp-py

# activate 'pomdp' environment by default
RUN echo "conda activate pomdp" >> ~/.bashrc