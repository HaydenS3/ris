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

For hyperparameter tuning, we'll take advantage of the RIS cluster's parallel processing capabilities, and we will do a grid search over the following hyperparameters. The list of hyperparameters can be easily modifed to scale the amount of compute needed.

#### Architecture

- **Number of Filters**: Number of filters in convolutional layers. [16, 32, 64, 128, 256]
  <!-- - **Filter Size**: Dimensions of the filters applied in convolutional layers. [3x3, 5x5] -->
  <!-- - **Number of Convolutional Layers**: Number of convolutional layers in the network. [2, 3, 4]
    <!-- - **Number of Convolutional Blocks**: A block consists of a convolutional layer followed by an activation function and pooling layer. [1, 2, 3] --> -->
    <!-- - **Activation Function**: Function applied to the output of each neuron, introducing non-linearity. [relu, sigmoid, tanh]
  <!-- - **Pooling Type**: Type of pooling operation (e.g., max pooling, average pooling). [MaxPooling2D, AveragePooling2D] -->
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

### [Compute 1](https://washu.atlassian.net/wiki/x/EgCSZw)

#### Access Information

- **Job Group**: /h.schroeder/ood
- **User Group**: compute-brianallen

#### Onboarding

Presentation given by Ayush Chaturvedi. There's multiple ways to access and run jobs on the RIS cluster. Would be good to try them all. Documentation on Confluence. Different batch operating systems to try. RIS is all in on Docker. They have a dedicated help desk.

#### [Job Groups](https://washu.atlassian.net/wiki/spaces/RUD/pages/1705182249/Job+Execution+Examples#Job-Groups)

Job groups limit jobs to N running jobs at a time and will queue jobs until resources are available.

**If you do not supply a job group with the bsub -g argument, a default job group named /${compute_username}/default will be created. This group has a default of 5 jobs, meaning only 5 jobs will run at a time, even if you are not at the max vCPU of 500.**

#### [Arrays](https://washu.atlassian.net/wiki/spaces/RUD/pages/1705182249/Job+Execution+Examples#Arrays)

When submitting a lot of similar jobs, they should submitted as an array. The maximum number of jobs in an array is 1000.

`bsub -J 'helloworld[1-5]' helloworld.sh \$LSB_JOBINDEX`. You have to specify a docker container too. `bsub -J 'helloworld[1-4]' -a 'docker(ubuntu:22.04)' /bin/echo \$LSB_JOBINDEX`

#### [Job Files](https://washu.atlassian.net/wiki/spaces/RUD/pages/1705182249/Job+Execution+Examples#bsub-Job-Files)

Nice way to store job configuration.

#### [Attach to Running Container](https://washu.atlassian.net/wiki/spaces/RUD/pages/1705182249/Job+Execution+Examples#Attach-to-an-Already-Running-Container)

Useful if you need to do something after a job has started.

#### [Parallel Jobs](https://washu.atlassian.net/wiki/x/SgCsZQ)

For running parallel jobs.

```
export LSF_DOCKER_NETWORK=host
export LSF_DOCKER_IPC=host
```

To test run: `bsub -G compute-brianallen -n 2 -R 'affinity[core(1)] span[ptile=1]' -I -q general-interactive -a 'docker(haydenschroeder/mpi-test)' mpirun -np 2 python3 /app/mpitest.py`

For ML project run: `bsub -G compute-brianallen -n 8 -R 'affinity[core(1)] span[ptile=1] rusage[mem=16GB]' -M 15GB -N -u h.schroeder@wustl.edu -q general -a 'docker(haydenschroeder/mpi-ml)' mpirun -np 8 python3 /app/rismpi.py`

For ML project on GPU run: `bsub -G compute-brianallen -R 'gpuhost' -gpu 'num=1:gmodel=NVIDIAA40:gmem=8G' -M 8GB -N -u h.schroeder@wustl.edu -q general -a 'docker(haydenschroeder/mpi-ml-gpu)' mpirun -np 1 python3 /app/rismpigpu.py`

#### [Real-Time Monitoring (RTM)](https://washu.atlassian.net/wiki/x/I4Fwag)

Real-time monitoring of jobs. Can be used to monitor job progress and resource usage.

#### [Software Development](https://washu.atlassian.net/wiki/x/HICCag)

How to develop software on the RIS cluster.

#### SSH

`ssh h.schroeder@compute1-client-3.ris.wustl.edu `

Open a basic interactive session with Python 3.10: `bsub -G compute-brianallen -Is -q general-interactive -a 'docker(python:3.10)' /bin/bash`

#### Open On Demand (OOD)

Jupyter Notebook. Works fine, seems like you can't run parallel jobs on OOD. You can run a single job, but it will be limited to the resources of the node you are on. This is good for testing and debugging, but not for running large jobs.

### [Compute 2](https://washu.atlassian.net/wiki/x/XwBRZw)

#### Access Information

- **User Group**: compute2-brainallen

#### Onboarding

Video recording of presentation. Communications and help happens via [the #compute2-early-adopters Slack channel](https://app.slack.com/client/TJGKAPR41/C08G3T0DPUY). **You need a paid SLack account to access the support channel.** Uses slurm scheduler instead of lsf. Documentation will probably be incomplete, still early access.

#### Transitioning

Use slurm instead of lsf; documentation is good for this and they've included a translator for the extremely lazy. Easier to run bare-metal jobs, however containers are certainly recommended. Compute2 is still early access; documentation is incomplete.

#### SSH

`ssh h.schroeder@c2-login-003.ris.wustl.edu`

Basicest test: `srun -p general-short /bin/bash helloworld.sh`

#### Arrays

`sbatch -p general-short --array=1-5 helloworld.sh`

#### [Using Containers](https://washu.atlassian.net/wiki/spaces/RUD/pages/1720615105/Compute2+Quickstart#Using-Containers)

Some documentation on using containers in compute2.

#### [Parallel Jobs](https://washu.atlassian.net/wiki/spaces/RUD/pages/2145517787/Compute2+MPI)

For test run make script:

```
#!/bin/bash
#SBATCH --partition=general-short
#SBATCH --nodes=2
#SBATCH --time=5
#SBATCH --container-image=haydenschroeder/mpi-test
#SBATCH --container-mounts=/cm,/etc/passwd,/lib64/libmunge.so.2,/run/munge,/storage2/fs1/brianallen/Active:/app/store
#SBATCH --container-env=PATH,SLURM_CONF
#SBATCH --container-writable
mpirun -np 2 python3 /app/mpitest.py
```

Then run: `sbatch mpitest.sh`

For ML project make script:

```
#!/bin/bash
#SBATCH --partition=general-cpu
#SBATCH --nodes=8
#SBATCH --time=120
#SBATCH --mem=16G
#SBATCH --container-image=haydenschroeder/mpi-ml
#SBATCH --container-mounts=/cm,/etc/passwd,/lib64/libmunge.so.2,/run/munge,/storage2/fs1/brianallen/Active:/app/store
#SBATCH --container-env=PATH,SLURM_CONF
#SBATCH --container-writable
mpirun -np 8 python3 /app/rismpi.py
```

Then run: `sbatch rismpi.sh`

For ML project big version make script:

```
#!/bin/bash
#SBATCH --partition=general-cpu
#SBATCH --nodes=16
#SBATCH --time=120
#SBATCH --mem=32G
#SBATCH --container-image=haydenschroeder/mpi-ml:big
#SBATCH --container-mounts=/cm,/etc/passwd,/lib64/libmunge.so.2,/run/munge,/storage2/fs1/brianallen/Active:/app/store
#SBATCH --container-env=PATH,SLURM_CONF
#SBATCH --container-writable
mpirun -np 16 python3 /app/rismpi.py
```

Then run: `sbatch rismpibig.sh`

For ML project on GPU make script:

```
#!/bin/bash
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --time=120
#SBATCH --gpus=1
#SBATCH --mem=4G
#SBATCH --container-image=haydenschroeder/mpi-ml-gpu
#SBATCH --container-mounts=/cm,/etc/passwd,/lib64/libmunge.so.2,/run/munge,/storage2/fs1/brianallen/Active:/app/store
#SBATCH --container-env=PATH,SLURM_CONF
#SBATCH --container-writable
mpirun python3 /app/rismpigpu.py
```

Then run: `sbatch rismpigpu.sh`

#### [Real-Time Monitoring (RTM)](https://washu.atlassian.net/wiki/spaces/RUD/pages/2145976581/Monitoring+Jobs+and+Partitions+Queues)

No nice user interface for compute2. Must use CLI.

#### [Open On Demand (OOD)](https://washu.atlassian.net/wiki/spaces/RUD/pages/2206924821/C2+Open+OnDemand+OOD)

Updated for compute2.

### Storage

Tried accessing storage through Globulus, but my permission was denied when loading `storage1` or `storage2` from the RIS Storage1 collection.

"In summary, a user must be a member of a Wash U AD group like storage-\* to see data via Globus. First a user should confirm access via SMB. If that works, the same storage volume should be visible in the Globus application." [Source](https://washu.atlassian.net/wiki/spaces/RUD/pages/1795948625/Storage1+Access+Control#Wash-U-Active-Directory-Groups)

`/storage1/fs1/brianallen/Active` and `/storage2/fs1/brianallen/Active`

#### Storage1

Unable to access.

[#### Storage2](https://washu.atlassian.net/wiki/x/H4ADaw)

Works great, able to upload files via Globus and view. Able to access it using jobs too: `srun -p general-interactive /bin/ls /storage2/fs1/brianallen/Active`

## Findings for Brian

- Learning compute1 took some time, mainly because I was learning how to build my own docker container, but wasn't too bad. Linux CLI experience helped a lot.
- Explored interactive sessions, job arrays, parallel jobs using MPI, open on demand, and real-time monitoring
- Used both CPU and GPU jobs. Show results
- Learning compute2 was super easy after learning compute1. The documentation is good and the transition from lsf to slurm is easy.
- RIS is all-in on Docker, but compute2 supports running bare-metal jobs
- Ran into issue on compute2 with MPI jobs. Unfortunately, I lost access to the support Slack channel yesterday. I have no idea what happened.
- Compute2 doesn't have a nice user interface for real-time monitoring, but the CLI works fine.
- Live demo if Brian wants any?
- Didn't use storage1 or storage2, but it seems pretty straightforward to use. Globus offers a nice GUI for transferring files.

## Meeting with Genomics Compute

### Recommendations

- Containers will make the transition to compute2 easier, less risk of breaking things. Compute2 does however support running bare-metal jobs.

### Questions

- Would like to understand differences between genomics compute and compute2.
  - Scheduler is slurm?
  - Containers?
  - Access to storage?
  - Running parallel jobs?
  - Monitoring jobs?
  - Interactive vs. batch jobs?
  - CPU vs. GPU?

## TODOs

- [ ] Run a BIG job. More hyperparameters, how many cores can I use, how much memory, how long does queue take?
- [ ] Run a GPU job
- [ ] Update script to output text file with best hyperparameters
- [ ] Use pytorch instead of keras.
- [ ] Add image augmentation to the model
- [ ] Improve CNN architecture. Add batch normalization layers to the model. Add early stopping to prevent overfitting.

```

```
