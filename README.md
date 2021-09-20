
# Debiased-CAM: Bias-agnostic faithful visual explanations for deep convolutional networks

### Introduction

This repository contains the Keras/Tensorflow implementation for the Debiased-CAM project. For more information, please check our paper at {paper link}. Please cite our paper if you find this code useful in your research:

	@Article{
	    To be updated
	}

### Environment

This code was tested on an Ubuntu 16.04 system using Tensorflow 2.1.0 and Keras 2.2.4.

### Datasets

The ImageNette dataset (with annotation) is available at https://github.com/fastai/imagenette. After downloading the ImageNette dataset, please put the train and val folders under ./datasets/imagenette/images/nobias/ to run the Debiased-CAM.

The NTCIR dataset is available at http://ntcir-lifelog.computing.dcu.ie/NTCIR12/. We use the visual activity recognition annotation provided by https://github.com/gorayni/egocentric_photostreams/tree/master/datasets/ntcir.

The COCO captioning dataset (with annotation) is available at https://cocodataset.org.


### Steps
1. preprocess: [Perturb (Bias) the Images](#Perturb)
2. train: [Train CNN models](#Train)
3. evaluate: [Evaluate CNN models](#Evaluate)

To check details in the manual for various options:
```sh
$ python main.py --help
```

##### Perturb (Bias) the Images
To apply the blur bias with sigma = 8 on the training set of ImageNette dataset. 
```sh
$ python main.py --step=preprocess --data_split=train --bias_type=blur --bias_level=8
```
To apply the color temperature bias with kelvin = -3600 on the training set of ImageNette dataset:
```sh
$ python main.py --step=preprocess --data_split=train --bias_type=ct --bias_level=-3600
```

##### Train CNN models
To train the RegularCNN with nobias images:
```sh
$ python main.py --step=train --bias_level=0 --model_type=regular 
```
To train the finetunedCNN_sb_st with bias=8:
```sh
$ python main.py --step=train --bias_level=8 --model_type=finetuned_sb_st 
```
To train the debiasedCNN_mb_mt with multibias images:
```sh
$ python main.py --step=train --bias_level=multibias --model_type=debiased_mb_mt 
```
To train the debiasedCNN_mb_st with bias=8:
```sh
$ python main.py --step=train --bias_level=8 --model_type=debiased_mb_st 
```

##### Evaluate CNN models
To evaluate the RegularCNN with nobias images on default tasks (performance and CAM faithfulness):
```sh
$ python main.py --step=evaluate --bias_level=0 --model_type=regular
```
To evaluate the debiasedCNN_mb_mt with multibias on default tasks (performance and CAM faithfulness):
```sh
$ python main.py --step=evaluate --bias_level=multibias --model_type=debiased_mb_mt
```
To evaluate the debiasedCNN_mb_st with bias=8 on default tasks (performance and CAM faithfulness):
```sh
$ python main.py --step=evaluate --bias_level=8 --model_type=debiased_mb_st
```
To evaluate the debiasedCNN_mb_mt on the bias regression task (with multibias images):
```sh
$ python main.py --step=evaluate --bias_level=multibias --model_type=debiased_mb_mt --eval_mode=regression
```

### License
The code and models in this repository are licensed under the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html) for academic and other non-commercial uses. For commercial use of the code and models, separate commercial licensing is available. Please contact:
- Brian Y. Lim ( brianlim@comp.nus.edu.sg )
- Jonathan Tan ( jonathan_tan@nus.edu.sg )
