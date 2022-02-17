FROM condaforge/miniforge3
WORKDIR /home

RUN apt-get --yes -qq update \
 && apt-get --yes -qq upgrade \
 && DEBIAN_FRONTEND=noninteractive \ 
 apt-get --yes -qq install \
                      build-essential \
                      cmake \
                      curl \
                      g++ \
                      gcc \
                      gfortran \
                      git \
                      libblas-dev \
                      liblapack-dev \
                      libopenmpi-dev \
                      openmpi-bin \
                      wget \
                      htop \
                      nano \
                      zsh \
 && apt-get --yes -qq clean \
 && rm -rf /var/lib/apt/lists/*
SHELL ["/bin/bash", "-c"]

RUN conda init bash
RUN conda init zsh

RUN conda create -n hmclab python==3.9
RUN echo "conda activate hmclab" >> $HOME/.zshrc
RUN echo "conda activate hmclab" >> $HOME/.bashrc
SHELL ["conda", "run", "-n", "hmclab", "/bin/bash", "-c"]

ADD hmclab-new-features /home/hmclab

RUN cd /home/hmclab && \
    pip install -e . 

RUN pip install tilemapbase
RUN conda install obspy numpy=1.21

RUN touch /home/start_server.sh && \
    echo "cd hmclab/notebooks && jupyter notebook --allow-root --no-browser --port=4324 --ip=0.0.0.0" \
    >> /home/start_server.sh && \
    chmod +x /home/start_server.sh 

CMD ["/bin/zsh" ]
