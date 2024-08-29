# syntax=docker/dockerfile:1

FROM tensorflow/tensorflow:2.17.0-gpu
RUN pip install grpcio==1.65.5
WORKDIR /splinter/communication
CMD ["python", "./split_computing_server.py"]
