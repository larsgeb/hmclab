FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y python3-pip python3-dev \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip

COPY . /hmc-tomography
WORKDIR /hmc-tomography
RUN pip install -v -e .[testing]
RUN pip install -v psvWave
RUN pip install jupyter

WORKDIR /hmc-tomography/examples/notebooks
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"] 