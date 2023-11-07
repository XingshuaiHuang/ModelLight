# ModelLight
ModelLight is a model-based meta-reinforcement learning framework for traffic signal control. 
This repository includes the detailed implementation of our ModelLight algorithm.
Usage and more information can be found below.

## Dataset
This repo contrains four real-world datasets (Hangzhou, Jinan, Atlanta, and Los Angeles) which are stored in
the `data.zip`. Please **unzip it** first before you run the code. More descriptions about the dataset can 
be found in the ModelLight paper.

## Installation Guide
### Dependencies
- python 3.*
- tensorflow v1.0+
- [cityflow](https://cityflow.readthedocs.io/en/latest/index.html)

### Quick Start 
We recommend to run the code through [docker](https://docs.docker.com/) and a sample docker image has been built for your quick start.

1. Please pull the docker image from the docker hub. 
``docker pull xsonline/modellight:latest``

2. Please run the built docker image, *xsonline/modellight:latest*, to initiate a docker container. Please remember to mount the code directory.
``docker run -it -v /local_path_to_the_code/modellight/:/modellight/ xsonline/modellight:latest /bin/bash``
Up to now, the experiment environment has been set up. Then, go the workspace in the docker contrainer, ``cd /modellight``, and try to run the code.



## Example 

Start an example:

``sh run_exp.sh``

It runs the file ``meta_train.py``. Here are some important arguments that can be modified for different experiments:

* memo: the memo name of the experiment
* algorithm: the specified algorithm, e.g., ModelLight, MetaLight, FRAPPlus. Please note that ModelLight is represented by MBMRL in our code.

Hyperparameters such as learning rate, sample size and the like for the agent can also be assigned in our code and they are easy to tune for better performance.
