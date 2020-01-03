# Tsinghua-Future-Internet-Project
In this project the team attempts to come up with an optimal solution to the problem at hand. The problem is to train a deep neural network with the VGG19 model on the CIFAR-19 image set consisting of 50 000 images for training and 10 000 images for validation. The images fall into 10 different classes that the model should be able to classify. The training will take place on a model that is not pretrained, initialized with random weights. Furthermore the problem can be split into two parts, the first one being successfully training the model, the second one being to utilize the hardware and network available at hand to the maximum efficiency. This paper proposes a parameter server style solution using: PyTorch and the adam optimizer. The team was unable to implement the proposed optimal solution, but still experiments with batch size and sees speedup results in the machine learning processing by utilizing the hardware in the distributed cluster more efficiently.



## Requirements

* Install PyTorch and Torchvision for CPU  ( from source ) 
* OpenMPI

## How to run

```bash 
mpirun --hostfile hosts -np 7 /home/a2019403475/.conda/havtob/bin/python3 Tsinghua-Future-Internet-Project/src/main/noe --epochs 100 --lr 0.001
```

