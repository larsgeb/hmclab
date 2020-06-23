# Use the official image as a parent image.
FROM python:3.7-slim

WORKDIR /home

RUN apt-get update 
RUN apt-get install git make pandoc -y

## Pass private key of host 

# parse argument
ARG SSH_KEY
ENV SSH_KEY=$SSH_KEY

# Make ssh dir
RUN mkdir /root/.ssh/

# Create id_rsa from string arg, and set permissions
RUN echo "$SSH_KEY" > /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa

# Create known_hosts
RUN touch /root/.ssh/known_hosts

# Add git providers to known_hosts
RUN ssh-keyscan bitbucket.org >> /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN ssh-keyscan gitlab.com >> /root/.ssh/known_hosts

RUN git clone git@github.com:larsgeb/hmc-tomography.git

WORKDIR /home/hmc-tomography
RUN pip install -e .[dev]