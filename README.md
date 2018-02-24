# Finetuning AlexNet for Oxford-102 Dataset

## Disclaimer
This is a work in progress, I am not responsible for any false information provided by the model.

## Cloning the project
To clone the project, use `git clone https://github.com/GhostGengar/AlexNet-Oxford102-Tensorflow.git`

## The dataset
Download the dataset from [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

You will need three files:
* [Dataset Images](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
* [The Image Labels](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat)
* [The data splits](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat)

*Please download the three required files and put them inside the `AlexNet` folder if you want to train using the `.py` file.*

## Start training
Run `python train.py` inside the AlexNet folder to start training the network.

## Start testing
This Tensorflow implementation comes with pretrained weights for Oxford-102, to download the pretrained weights:
* [The `.data` file](https://drive.google.com/file/d/1cibsejSXefO8rb-tZAPimULkjXkKzkQI)
* [The `.index` file](https://drive.google.com/file/d/1Bjligj4trKPIL1YV9naorEy7S1PD9wzO)
* [The `.meta` file](https://drive.google.com/open?id=1Ll4PERfHF_G5HJX00nGqs1gmT8RFPEIR)

*Download all three files and put them inside a new folder named `models` inside the `AlexNet` folder.*

Pick an image of your choice and put it inside the `images` folder, rename it to `flower.jpg`.

To start training, run `python flower_test.py`

## Requirements
* numpy
* cv2
* scipy
* tensorflow

## Explaination
* Full code description is available inside the `Oxford_102_AlexNet.ipynb` file.
