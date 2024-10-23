#! /bin/bash
g5k-setup-docker -t
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo-g5k apt-get update
sudo-g5k apt-get install -y nvidia-container-toolkit
sudo-g5k systemctl restart docker
sudo-g5k dhclient -6 br0
cd splinter
docker build -t splinter .
docker run --gpus all --rm -v /home/dmay/splinter:/splinter -p 50051:50051 -e "HOSTNAME=$(cat /etc/hostname)" splinter