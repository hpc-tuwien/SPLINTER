# syntax=docker/dockerfile:1

FROM tensorflow/tensorflow:2.15.0-gpu
RUN pip install grpcio==1.66.1
RUN pip install pandas==2.2.2
RUN pip install psutil==6.0.0
RUN pip install nvidia-ml-py==12.560.30
RUN pip install scikit-learn==1.5.1
WORKDIR /splinter/communication
CMD ["python", "./split_computing_server.py"]
