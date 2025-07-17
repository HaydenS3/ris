# RIS Project

Author: Hayden Schroeder

## Overview

Need to test the RIS cluster and get a better understanding of how it works. I plan to use [Kaggle's Digit Recognizer challenge](https://www.kaggle.com/competitions/digit-recognizer/overview) to test the RIS cluster. I see two opportunities for parallelization using MPI:

1. **Hyperparameter Tuning:** Generate 300+ hyperparameter combinations and run the jobs separately to find hyperparameters with the best performance. Going to pick somewhere between
2. **Data Preprocessing:** Split up the dataset to perform preprocessing across multiple jobs.

Use [MPI for Python](https://mpi4py.readthedocs.io/en/stable/overview.html) to implement the parallelization. The goal is to run the jobs on the RIS cluster and compare the performance with a single job run on a local machine. Further, here is a [parallel programming course which uses MPI](https://theartofhpc.com/pcse/index.html).

## Codebase

### Tutorial

https://www.kaggle.com/code/poonaml/deep-neural-network-keras-way/notebook

### Installation

```
conda create -n ris python=3.10
conda activate ris
conda install --file requirements.txt
conda install -c conda-forge mpi4py openmpi
```

### Running the Code

First run the test MPI job using `mpirun -np 10 python mpitest.py`. This will run the code on 10 processes. The code will generate a random number for each process and print it out. This is a good way to test that MPI is working correctly. This will fail if the computer has less than 10 cores.

If the test job works, you can run the main job using `mpirun -np <number_of_processes> python rismpi.py`. This will run the code on the specified number of processes.

### CNN Hyperparameters

For hyperparameter tuning, we'll take advantage of the RIS cluster's parallel processing capabilities, and we will do a grid search over the following hyperparameters. The current list of hyperparameters generates a total of 13,824 combinations.

#### Architecture

- **Number of Filters**: Number of filters in convolutional layers. [16, 32, 64, 128, 256]
    <!-- - **Filter Size**: Dimensions of the filters applied in convolutional layers. [3x3, 5x5] -->
    <!-- - **Number of Convolutional Layers**: Number of convolutional layers in the network. [2, 3, 4]
  <!-- - **Number of Convolutional Blocks**: A block consists of a convolutional layer followed by an activation function and pooling layer. [1, 2, 3] --> -->
  <!-- - **Activation Function**: Function applied to the output of each neuron, introducing non-linearity. [relu, sigmoid, tanh]
- **Pooling Type**: Type of pooling operation (e.g., max pooling, average pooling). [MaxPooling2D, AveragePooling2D] -->
- **Number of Neurons in Dense Layers**: Number of neurons in fully connected layers. [64, 128, 256, 512]
  <!-- - **Number of Dense Layers**: Number of fully connected layers in the network. [1, 2, 3] -->
  <!-- - **Optimizer**: Algorithm used to update the weights of the network based on the loss function (e.g., Adam, SGD). [Adam, SGD, RMSprop] -->

#### Other

- **Batch Size**: Number of training examples utilized in one iteration. [32, 64, 128, 256]
<!-- - **Epochs**: Number of complete passes through the training dataset. [5, 10, 15, 20] -->
- **Learning Rate**: Step size at each iteration while moving toward a minimum of the loss function. A high learning rate can lead to faster convergence but may overshoot the minimum. [0.001, 0.01, 0.1]
- **Dropout Rate**: Fraction of the input units to drop during training to prevent overfitting. [0.2, 0.3, 0.4, 0.5]
- **Validation Split**: Fraction of the training data to be used as validation data. [0.1, 0.2, 0.3]

### Reading & Sources (in order of complexity?)

- [Neural Networks - 3blue1brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi): If you prefer video explanations.
- [Convolutional Neural Network from Scratch](https://medium.com/latinxinai/convolutional-neural-network-from-scratch-6b1c856e1c07)
- [Understanding of Convolutional Neural Network](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148)
- [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/): This one includes great visualizations including those from the MNIST dataset.
- [Stanford Convolutional Neural Networks course](https://cs231n.github.io/convolutional-networks/): This one really gets into the details. Really great information on how to design CNN architectures.
- [Backpropogation and Gradient Descent IBM](https://www.ibm.com/think/topics/backpropagation)
- “Deep Learning” by Ian Goodfellow, Yoshua Bengio, and Aaron Courville — A comprehensive textbook covering all aspects of deep learning.
- “Neural Networks and Deep Learning” by Michael Nielsen — An accessible introduction to neural networks and deep learning.
- Research Paper: “ImageNet Classification with Deep Convolutional Neural Networks” by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton — The groundbreaking paper that introduced AlexNet.

### Notes

The main reason for using CNNs is that regular neural networks don't scale well to image data due to the high dimensionality of images.

#### CNN Layers

- **Convolutional Layers**: Detect features in input images by applying filters. Generally, multiple layers are stacked and multiple filters are applied. []
- **Pooling Layers**: Reduce dimensionality and retain important features.
  - Max Pooling vs. Average Pooling: Max pooling is generally preferred as it retains the most significant features.
- **Fully Connected Layers**: Connect every neuron in one layer to every neuron in the next layer, typically used at the end of the network.
- **Activation Functions**: Introduce non-linearity to the model, commonly using ReLU (Rectified Linear Unit) for hidden layers and softmax for output layers in classification tasks.
- **Dropout Layers**: Prevent overfitting by randomly setting a fraction of input units to 0 during training.
- **Batch Normalization Layers**: Normalize the inputs to each layer to improve training speed and stability.
- **Flatten Layers**: Convert multi-dimensional input into a one-dimensional vector, typically used before fully connected layers.
- **Loss Function**: Measures the difference between predicted and actual values, commonly using categorical crossentropy for multi-class classification tasks.

#### Architecture

- A lot of great information in [Stanford CS231n](https://cs231n.github.io/convolutional-networks/).
- "If you’re feeling a bit of a fatigue in thinking about the architectural decisions, you’ll be pleased to know that in 90% or more of applications you should not have to worry about these. I like to summarize this point as “don’t be a hero”: Instead of rolling your own architecture for a problem, you should look at whatever architecture currently works best on ImageNet, download a pretrained model and finetune it on your data." (Stanford CS231n)

## RIS

### Onboarding

Presentation given by Ayush Chaturvedi. There's multiple ways to access and run jobs on the RIS cluster. Would be good to try them all. Documentation on Confluence. Different batch operating systems to try. RIS is all in on Docker. They have a dedicated help desk.

### Access Information

- **Job Group**: /h.schroeder/ood
- **User Group**: compute-brianallen

### Compute 1

#### [Job Groups](https://washu.atlassian.net/wiki/spaces/RUD/pages/1705182249/Job+Execution+Examples#Job-Groups)

Job groups limit jobs to N running jobs at a time and will queue jobs until resources are available.

**If you do not supply a job group with the bsub -g argument, a default job group named /${compute_username}/default will be created. This group has a default of 5 jobs, meaning only 5 jobs will run at a time, even if you are not at the max vCPU of 500.**

#### [Arrays](https://washu.atlassian.net/wiki/spaces/RUD/pages/1705182249/Job+Execution+Examples#Arrays)

When submitting a lot of similar jobs, they should submitted as an array. The maximum number of jobs in an array is 1000.

#### [Job Files](https://washu.atlassian.net/wiki/spaces/RUD/pages/1705182249/Job+Execution+Examples#bsub-Job-Files)

Nice way to store job configuration.

#### [Attach to Running Container](https://washu.atlassian.net/wiki/spaces/RUD/pages/1705182249/Job+Execution+Examples#Attach-to-an-Already-Running-Container)

Useful if you need to do something after a job has started.

#### [Parallel Jobs](https://washu.atlassian.net/wiki/x/SgCsZQ)

For running parallel jobs.

```
export LSF_DOCKER_NETWORK=host
export LSF_DOCKER_IPC=host
bsub -G compute-brianallen -n 10 -R 'affinity[core(1)] span[ptile=1]' -I -q general-interactive -a 'docker(haydenschroeder/openmpi-python:3.10)'
```

#### [Real-Time Monitoring (RTM)](https://washu.atlassian.net/wiki/x/I4Fwag)

Real-time monitoring of jobs. Can be used to monitor job progress and resource usage.

#### [Software Development](https://washu.atlassian.net/wiki/x/HICCag)

How to develop software on the RIS cluster.

#### SSH

Open a basic interactive session with Python 3.10: `bsub -G compute-brianallen -Is -q general-interactive -a 'docker(python:3.10)' /bin/bash`

#### Open On Demand (OOD)

Jupyter Notebook. Works fine, seems like you can't run parallel jobs on OOD. You can run a single job, but it will be limited to the resources of the node you are on. This is good for testing and debugging, but not for running large jobs.

### Storage

Tried accessing storage through Globulus, but my permission was denied when loading `storage1` or `storage2` from the RIS Storage1 collection.

"In summary, a user must be a member of a Wash U AD group like storage-\* to see data via Globus. First a user should confirm access via SMB. If that works, the same storage volume should be visible in the Globus application." [Source](https://washu.atlassian.net/wiki/spaces/RUD/pages/1795948625/Storage1+Access+Control#Wash-U-Active-Directory-Groups)

#### Storage1

#### Storage2

## TODOs

- [ ] Try parallel computing
- [ ] Try job groups and arrays
- [ ] Try multiple different configuration options
  - [ ] Specify specific CPUs and memory limits
- [ ] Try compute2
- [ ] Try other things mentioned?
- [ ] Add log file so I can see progress without interactive shell?
- [ ] Create script version for GPU jobs?
- [ ] Use pytorch instead of keras.
- [ ] Add image augmentation to the model
- [ ] Improve CNN architecture. Add batch normalization layers to the model. Add early stopping to prevent overfitting.
