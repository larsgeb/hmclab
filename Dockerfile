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

RUN mkdir /home/hmclab
ADD .  /home/hmclab

RUN cd /home/hmclab && \
    pip install -e . 

RUN pip install psvWave==0.2.1


CMD ["conda", "run", "--no-capture-output", "-n", "hmclab", "jupyter", \
     "notebook", "--notebook-dir=hmclab/notebooks", "--ip=0.0.0.0", \
     "--port=8888", "--allow-root", "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
