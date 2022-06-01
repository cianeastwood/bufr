# Bottom-Up Feature Restoration

This repository contains the code accompanying the paper [Source-Free Adaptation to
Measurement Shift via Bottom-Up Feature Restoration](https://arxiv.org/pdf/2107.05446.pdf) (ICLR 2022, Spotlight).

### Prerequisites:
- python 3
- pytorch 1.7.0+
- [WILDS](https://github.com/p-lambda/wilds) (for Camelyon-17)
- numpy, scipy, sklearn, matplotlib, PIL, torchvision 

## Data
By default datasets are assumed to be located in a ./datasets/ directory, this can be changed with 
the data-root flag in the shell scripts used to run experiments. The table below shows where the datasets can be 
downloaded from and the path where they should be stored. To start running the code only EMNIST-DA should be required.

| Dataset     | Download                                                                                                                          | Location                    |
|-------------|-----------------------------------------------------------------------------------------------------------------------------------|-----------------------------|
| EMNIST-DA   | [Data](https://zenodo.org/record/6602351)                                        | &lt;data-root>/EMNIST/      |
| MNIST-C     | [Data](https://zenodo.org/record/3239543)                                                                                         | &lt;data-root>/mnist_c/     |
| MNIST-M     | Set download=True in mnistm.py or [here](https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz) | &lt;data-root>/MNISTM/      |
| CIFAR-10-C  | [Data](https://zenodo.org/record/2535967)                                                                                         | &lt;data-root>/CIFAR-10-C/  |
| CIFAR-100-C | [Data](https://zenodo.org/record/3555552)                                                                                         | &lt;data-root>/CIFAR-100-C/ |
| CAMELYON-17 | Downloads automatically (download=True) from WILDS                                                                                | &lt;data-root>/CAMELYON17/  |

## Simple Usage

To run Bottom-Up Feature Restoration there are three steps: 

1. Pretrain the source model
```bash
./scripts/pretrain.sh
```
2. Save the source statistics
```bash
./scripts/save_train_stats.sh
```

3. Adapt on the new data
```bash
./scripts/bufr.sh
```

Other (not bottom-up) methods, including source-only performance, AdaBN and feature restoration can be run after steps 
1 & 2 by runnning
```bash
./scripts/adapt.sh
```

## Further Information

#### Choosing what to run
In the configs directory are yaml files that set hyperparameters for the different methods for each dataset
- ```configs/<DATASET_NAME>/dataset.yml``` sets arguments used for all methods across a dataset, namely the network 
architecture and which shifts to train on.

- ```configs/<DATASET_NAME>/<method>.yml``` sets the specific hyperparameters for the different methods
(learning rate, epochs etc.)

The shell scripts in the scripts directory set which method to run, which dataset to use and some global arguments. To 
choose a specific dataset the following arguments should be set where applicable
- ```--alg-config``` to ```./configs/<MDATASET_NAME>/pretrain.yml```
- ```--alg-configs-dir``` to ```./configs/<DATASET_NAME>/```
- ```--data-config``` to` ```./configs/<DATASET_NAME>/dataset.yml```

Possible values for ```<DATASET_NAME>``` are: ```EMNIST-DA```, ```MNIST-C```, ```CIFAR-10-C```, ```CIFAR-100-C```, 
```CAMELYON17```. (The directory names inside configs.)

To choose a specific method in ```scripts/adapt.sh``` set ```--alg-name``` to the name of the method. Possible values 
for ```--alg-name``` are: 
- ```all``` - trains all methods for the dataset
- ```fewshot``` - for EMNIST-DA and CAMELYON17 trains the few-shot methods
- Any one of: ```adabn```, ```bnm```, ```bnm_im```, ```fr```, ```im```, ```jg```, ```label```, ```pl```, ```shot```, 
```source_only```. (The names of the yaml files inside
```configs/<DATASET_NAME>/```. Not all methods are implemented for all datasets.)

The same random seed should be used for all steps (pretraining, saving statistics and adapting), this can be set with 
the ```--seed``` flag.






#### SHOT baseline
[SHOT](https://github.com/tim-learn/SHOT/) uses different pretraining techniques, to run this baseline on EMNIST-DA or 
MNIST-C requires first rerunning ./scripts/pretrain.sh on with the ```shot_pretrain``` argument set to True in 
pretrain.yml.


#### Analyses
The analysis directory contains files for creating analysis plots and calculating the feature restoration score. These 
analyses require an adapted models to be saved in advance, this can be done by adding ```--save-adapted-model``` to the
adapt.sh and bufr.sh scripts during training. 

Analysis files can be run using the scripts in ```./scripts/analysis``` and changing the PYTHONPATH appropriately. 
The files calibration.py, max_patches.py and tsne.py contain hardcoded paths at the start of the files that you may 
wish to change.

## Results
The results we achieved from running this code can be found [here](Results.md).

## Citation
```
@inproceedings{eastwood2022sourcefree,
     title={Source-Free Adaptation to Measurement Shift via Bottom-Up Feature Restoration}, 
     author={Cian Eastwood and Ian Mason and Christopher K. I. Williams and Bernhard Sch\"olkopf},
     booktitle={International Conference on Learning Representations},
     year={2022}
}
```

Readers may also be interested in our related study on [Unit-level surprise in neural networks](https://www.ianxmason.com/papers/unit_level_surprise.pdf).
