# syntax=docker/dockerfile:1

FROM tensorflow/tensorflow:2.15.0-gpu
WORKDIR /splinter/communication
CMD ["python", "./split_computing_server.py"]
