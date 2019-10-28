# FIIT Neural Networks Project

This project is elaborated as an assignment for the [Neural Networks course](https://github.com/matus-pikuliak/neural_networks_at_fiit) (WS 2019/2020) at FIIT STU by [Denis Mitana](https://github.com/dmitana/) and [Miroslav Sumega](https://github.com/xsumegam/).

## Installation
1. Build docker image
```sh
$ cd docker
$ docker build -t fiit_nn_project/tensorflow:2.0.0-gpu-py3-jupyter .
```

2. Start docker container
```sh
$ cd ..
$ docker run --gpus all -u $(id -u):$(id -g) -p 8888:8888 -p 6006:6006 -v $(pwd):/tf/fiit_nn_project -it --name fiit_nn_project fiit_nn_project/tensorflow:2.0.0-gpu-py3-jupyter
```
