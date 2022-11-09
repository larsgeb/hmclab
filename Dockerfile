FROM jupyter/datascience-notebook
WORKDIR /home/jovyan/
RUN mkdir /home/jovyan/hmclab

# Add Python files from this project
ADD --chown=jovyan:users . /home/jovyan/hmclab

# Install required prerequisites for Python version.
RUN pip install -e /home/jovyan/hmclab
RUN pip install psvWave==0.2.1

# Change the startup location of the notebook server.
WORKDIR /home/jovyan/hmclab/notebooks/

