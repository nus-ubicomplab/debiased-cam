"""
This file support perturbing the dataset, training and evaluating CNNs. e.g. 


To apply the blur bias with sigma = 8 on the training set of ImageNette dataset:

    python main.py --step=preprocess --data_split=train --bias_type=blur --bias_level=8

To apply the color temperature bias with kelvin = -3600 on the training set of ImageNette dataset:

    python main.py --step=preprocess --data_split=train --bias_type=ct --bias_level=-3600


To train the RegularCNN with nobias images:
    
    python main.py --step=train --bias_level=0 --model_type=regular 

To train the FinetunedCNN_sb_st with bias=8:
    
    python main.py --step=train --bias_level=8 --model_type=finetune_sb_st 

To train the DebiasedCNN_mb_mt with multibias images:
    
    python main.py --step=train --bias_level=multibias --model_type=debias_mb_mt 


To evaluate the RegularCNN with nobias images on default tasks (performance and CAM faithfulness):
    
    python main.py --step=evaluate --bias_level=0 --model_type=regular

To evaluate the DebiasedCNN_mb_mt with bias=8 on default tasks (performance and CAM faithfulness):
    
    python main.py --step=evaluate --bias_level=8 --model_type=debiased_mb_mt

To evaluate the DebiasedCNN_mb_mt on the bias regression task (with multibias images):
    
    python main.py --step=evaluate --bias_level=multibias --model_type=debiased_mb_mt --eval_mode=regression


You can check more details in the manual:: 
    
    python main.py --help

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import sys
import time
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_control_flow_v2()
from tensorflow.keras.utils import to_categorical

from perturb_bias import create_clear_annotation, perturb_blur_multi_bias, perturb_blur_specific_bias, perturb_color_multi_bias, perturb_color_specific_bias
from train import train_finetune, train_debias, train_siamese, get_iterator, batch_generator_finetune, N_CLASSES, CAM_WIDTH_MAP, IMG_WIDTH_MAP, num_validation_samples, class_weight
from evaluate import make_prediction, make_prediction_siamese, eval_label_global, eval_label_confidence, eval_heatmap, eval_bias_regress

MODEL_PREFIX_MAP = {'debiased_sb_mt':"debias", 'debiased_mb_mt':"debias", 'debiased_sb_st':"siamese", 'debiased_mb_st':"siamese", 'regular':"finetune", 'finetuned_sb_st':"finetune", 'finetuned_mb_st':"finetune"}

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare the biased image data')
    parser.add_argument('--step', dest='step',
                        help='Steps of the processing pipeline:\n preprocess (Perturb/Bias the Images),  train (Train CNN models),  evaluate (Evaluate CNN models)',
                        default='preprocess', type=str)
    parser.add_argument('--bias_type', dest='bias_type',
                        help='Data bias type. Possible options: blur or ct',
                        default="blur", type=str)
    parser.add_argument('--bias_level', dest='bias_level',
                        help='Data bias level. 0 for nobias.\n Gaussian kernel size for blur (possible values: 2, 8, 16, 32) \n Kelvin for color temperature (ct) (possible values: -3600, -1800, +1800, -3600)\n "multibias" to train across all bias levels',
                        default='0', type=str)
    parser.add_argument('--data_split', dest='data_split',
                        help='Only specified for the preprocess step.\n Possible options: train, val(test)',
                        default='train', type=str)
    parser.add_argument('--base_model', dest='base_model',
                        help='Base CNN model used in the training step, or the evaluation step',
                        default='InceptionV3', type=str)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='Batch size used in the training step, or the evaluation step',
                        default=64, type=int)
    parser.add_argument('--model_type', dest='model_type',
                        help='The model type for training or evaluation. Possible options:\n regular, finetuned_sb_st, finetuned_mb_st, debiased_sb_mt, debiased_mb_mt, debiased_sb_st, debiased_mb_st',
                        default='regular', type=str)
    parser.add_argument('--eval_mode', dest='eval_mode',
                        help='The evaluation mode. Possible options:\n default (Task Performance & CAM faithfulness), regression (for bias level),',
                        default='default', type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()    
    model_params_dir = './model_params'

    # To get the nobias annotation frames
    if not os.path.exists("./datasets/imagenette/nobias_train.csv"):
        create_clear_annotation("train")
    if not os.path.exists("./datasets/imagenette/nobias_val.csv"):
        create_clear_annotation("val")

    # Perturb (bias) the dataset
    if args.step == 'preprocess':
        print("Perturb (bias) the dataset ")
        print("Data bias type is: "+ args.bias_type)
        print("Data bias level is: "+ args.bias_level)
        print("Dataset split is: "+args.data_split)
        if 'blur' in args.bias_type:
            if 'multibias' in args.bias_level:
                perturb_blur_multi_bias(args.data_split)
            else:
                perturb_blur_specific_bias(args.data_split, args.bias_level)
        elif 'ct' in args.bias_type:
            if 'multibias' in args.bias_level:
                perturb_color_multi_bias(args.data_split)
            else:
                perturb_color_specific_bias(args.data_split, args.bias_level)

    # Train CNN models          
    elif args.step == 'train':
        print("Training mode is: "+ args.model_type)
        print("Training (bias) data type is: "+ args.bias_type)
        print("Training (bias) data level is: "+ args.bias_level)
        time_start = time.time()
        if 'debiased' not in args.model_type:
            train_finetune(args.bias_type, args.bias_level, model_params_dir, args.batch_size, 20, args.base_model)
        else:
            if 'mt' in args.model_type:
                train_debias(args.bias_type, args.bias_level, model_params_dir, args.batch_size, 20, args.base_model)  
            else:
                train_siamese(args.bias_type, args.bias_level, model_params_dir, args.batch_size, 20, args.base_model)  
        print('Training done, with the time cost: ',time.time() - time_start)

    # Evaluate CNN models
    elif args.step == 'evaluate':
        print("Evaluation type is: "+ args.eval_mode)
        print("Evaluation model is: "+ args.model_type)
        print("Evaluation (bias) data type is: "+ args.bias_type)
        print("Evaluation (bias) data level is: "+ args.bias_level)
        data_type = "nobias" if args.bias_level == '0' else args.bias_type+'_'+args.bias_level
        model_prefix = MODEL_PREFIX_MAP[args.model_type]
        target_model_dir = '{}/{}_{}_{}.hdf5'.format(model_params_dir, args.base_model, model_prefix, data_type)
    
        img_sz = IMG_WIDTH_MAP[args.base_model]
        heatmap_sz = CAM_WIDTH_MAP[args.base_model]

        # To evaluate the bias regression
        if args.eval_mode == 'regression':
            iterator_br = get_iterator("./datasets/imagenette/{}_multibias_val.csv".format(args.bias_type), 1, args.batch_size, (img_sz, img_sz))
            gt_br = iterator_br.labels[:num_validation_samples].astype("float32")
    
        # To evaluate classification and CAM faithfulness
        iterator_cls = get_iterator("./datasets/imagenette/nobias_val.csv", 0, args.batch_size, (img_sz, img_sz))
        gt_cls = to_categorical(iterator_cls.labels[:num_validation_samples], N_CLASSES)
        filenames = iterator_cls.filenames[:num_validation_samples]
        test_generator = batch_generator_finetune("./datasets/imagenette/{}_val.csv".format(data_type), args.batch_size)
        class_weight = class_weight / np.sum(class_weight)

        time_start = time.time()
        if "siamese" not in model_prefix:
            pred_cls, pred_heatmap, pred_br = make_prediction(target_model_dir, test_generator, num_validation_samples, args.batch_size)
        else:
            pred_cls, pred_heatmap, pred_br = make_prediction_siamese(target_model_dir, test_generator, num_validation_samples, args.batch_size)
        np.save("./heatmap_npy/{}_{}_{}_{}.npy".format(args.base_model, args.bias_type, model_prefix, data_type), pred_heatmap)
        
        if args.eval_mode == 'regression':
            # Only meaningful to evaluate the bias regression for multi-bias CNNs
            if args.bias_level == 'multibias' and 'mb' in args.model_type:
                eval_bias_regress(gt_bl, pred_br, args.model_type, args.bias_type, args.bias_level, filenames)
        else:
            eval_label_global(gt_cls, pred_cls, args.model_type, args.bias_type, args.bias_level, cls_weights)
            eval_label_confidence(gt_cls, pred_cls, args.model_type, args.bias_type, args.bias_level, filenames)
            if args.model_type == 'regular' and args.bias_level == '0':
                # No need to evalute the heatmap for Regular CNN with unbiased images
                pass
            else:
                # Need to evaluate Regular CNN with nobias test set before evaluating other heatmaps
                ref_heatmap = np.load("./heatmap_npy/{}_{}_finetune_nobias.npy".format(args.base_model, args.bias_type))
                eval_heatmap(ref_heatmap, pred_heatmap, args.model_type, args.bias_type, args.bias_level, filenames)
        print('Evaluation done, with the time cost: ',time.time() - time_start)
