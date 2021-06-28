
FROM continuumio/miniconda3:4.7.12


ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install unzip

RUN conda create -n p37 -c conda-forge python=3.7 fenics
ENV PATH /opt/conda/envs/p37/bin:$PATH
RUN /bin/bash -c "source activate p37"
RUN conda install -c anaconda hdf5=1.12.0
RUN which python
RUN wget https://zenodo.org/record/3634136/files/hippylib/hippylib-3.0.0.zip
RUN unzip hippylib-3.0.0.zip
WORKDIR hippylib-hippylib-0402982/
RUN pip install -e .

COPY . ~/hmc-tomography
WORKDIR ~/hmc-tomography

RUN python3.7 -m pip install -e .[testing]
#RUN python3.7 -m pip install -v psvWave
RUN python3.7 -m pip install install jupyter

ARG DEBIAN_FRONTEND=dialog

#ENV HDF5_DISABLE_VERSION_CHECK 2

WORKDIR /hmc-tomography/examples/notebooks
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"] 
