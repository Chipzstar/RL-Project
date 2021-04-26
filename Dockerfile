FROM ubuntu

# Install apt-get in Docker

RUN apt-get update
RUN mkdir -p /testdocker
RUN apt-get install -y software-properties-common

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip


# Install libraries

RUN pip3 install jupyter


# Push current directory to docker

COPY . ./testdocker

# Install Jupyter notebook on docker

RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888
CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/testdocker", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
