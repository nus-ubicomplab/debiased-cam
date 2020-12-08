'''

This script contains several evaluation functions 

- make_prediction: get the inference from CNN models 
        regular, finetuned_sb_st, finetuned_mb_st, debiased_sb_mt, debiased_mb_mt, 

- make_prediction_siamese: get the inference from CNN models with siamese structure
        debiased_sb_st, debiased_mb_st

- eval_label_global: evaluate the model's performance, e.g. accuracy, f1-score, prauc

- eval_heatmap: evaluate the model's CAM faithfulness performance

- eval_bias_regress: evaluate the model's bias regression performance

- eval_label_confidence: evaluate model's the confidence score

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import time
import sklearn
import metrics
import argparse

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
from train import get_iterator, init_multitask_cnn, init_siamese_model, N_CLASSES

def make_prediction(model_dir, generator, num_test_samples, batch_size=16, base_model='InceptionV3'):
    # Load the multi-task CNN model
    model = init_multitask_cnn(base_model)
    model.load_weights(model_dir, by_name=True)
    pred_cls, pred_heatmap, pred_biaslevel = model.predict(generator, steps=num_test_samples//batch_size, verbose=1)
    return pred_cls, pred_heatmap, pred_biaslevel

def get_debias_branch(model):
    model = Model(inputs=[model.inputs[0], model.inputs[1]], outputs=[model.outputs[0], model.outputs[1], model.outputs[2]])   
    return model

def make_prediction_siamese(model_dir, generator, num_test_samples, batch_size=16, base_model='InceptionV3'):
    # Load the two-branch siamese model, but only use the debias-branch for intefence
    weights_dir = model_dir.split('/')[0]
    bias_type = model_dir.split('/')[1].split('_')[1]
    siamese_model = init_siamese_model(weights_dir, bias_type, False)
    siamese_model.load_weights(model_dir, by_name=True)
    debias_model = get_debias_branch(siamese_model)
    pred_cls, pred_heatmap, pred_biaslevel = debias_model.predict(generator, steps=num_test_samples//batch_size, verbose=1)
    return pred_cls, pred_heatmap, pred_biaslevel

def eval_label_global(gt_prob, pred_prob, model_type, bias_type, bias_level, weight, base_model='InceptionV3'):
    # Evaulate the labeling performance globally.
    # Metrics considered: Accuracy, Precision, Recall, F1-score, PRAUC
    data_type = "nobias" if bias_level == '0' else bias_type+'_'+bias_level
    gt_label = np.argmax(gt_prob, axis=1)
    pred_label = np.argmax(pred_prob, axis=1)
    accuracy = accuracy_score(gt_label, pred_label)

    # Use weighted_macro considering the unbalanced dataset
    weighted_precision = precision_score(gt_label, pred_label, average='weighted')
    weighted_recall = recall_score(gt_label, pred_label, average='weighted')
    weighted_f1 = f1_score(gt_label, pred_label, average='weighted')
    precision = precision_score(gt_label, pred_label, average=None)
    recall = recall_score(gt_label, pred_label, average=None)
    f1 = f1_score(gt_label, pred_label, average=None)
    pr_auc = list()
    precision_curve = dict()
    recall_curve = dict()
    for i in range(N_CLASSES):
        precision_curve[i], recall_curve[i], _ = precision_recall_curve(gt_prob[:, i], pred_prob[:, i])
        pr_auc.append(auc(recall_curve[i], precision_curve[i]))
    pr_auc = np.array(pr_auc)
    weighted_prAuc = np.nansum(weight * pr_auc)
    macro_prAuc = np.nanmean(pr_auc)

    label_summary = pd.DataFrame([[model_type+'_'+data_type, "weighted", accuracy, weighted_precision, weighted_recall, weighted_f1, weighted_prAuc]])
    label_summary.to_csv('eval_results/label_summary_{}.csv'.format(bias_type), header=False, mode='a+')
    return

def eval_heatmap(gt_heatmap, bias_heatmap, model_type, bias_type, bias_level, filenames, base_model='InceptionV3'):
    # Evaulate the heatmap faithfulness by instance.
    # Metrics considered: L2 distance, PCC distance, JS divergence, SIM distance
    data_type = "nobias" if bias_level == '0' else bias_type+'_'+bias_level
    dist_0 = metrics.distL2(gt_heatmap, bias_heatmap)
    dist_1 = metrics.distPCC(gt_heatmap, bias_heatmap)
    dist_2 = metrics.distJS(gt_heatmap, bias_heatmap)
    dist_3 = metrics.distSIM(gt_heatmap, bias_heatmap)
    dist_detail = pd.DataFrame({"distMSE":dist_0, "distPCC":dist_1, "distJS":dist_2, "distSIM":dist_3, "filenames":filenames})
    dist_detail.to_csv('eval_results/heatmap_{}_{}.csv'.format(model_type, data_type), mode='w+')
    return 

def eval_bias_regress(gt_bl, pred_bl, model_type, bias_type, bias_level, filenames, base_model='InceptionV3'):
    # Evaulate the bias level regression by instance.
    # Note: only works for multibias data
    data_type = "nobias" if bias_level == '0' else bias_type+'_'+bias_level
    r_square = metrics.get_r_square(gt_bl, pred_bl)         
    results = pd.DataFrame({'gt':np.squeeze(gt_bl), 'pred':np.squeeze(pred_bl), 'filenames': filenames})
    results.to_csv("eval_results/bias_regress_{}_{}.csv".format(model_type, data_type), mode='w+')
    return

def eval_label_confidence(gt_cls, pred_cls, model_type, bias_type, bias_level, filenames, base_model='InceptionV3'):
    # Evaulate the labeling confidence score by instance.
    data_type = "nobias" if bias_level == '0' else bias_type+'_'+bias_level
    gt_label = np.squeeze(np.argmax(gt_cls, axis=1))
    pred_label = np.squeeze(np.argmax(pred_cls, axis=1))
    score_pred = [pred_cls[i, pred_label[i]] for i in range(len(pred_label))] 
    score_gt = [pred_cls[i, gt_label[i]] for i in range(len(pred_label))] 
    results = pd.DataFrame({'gt_class':gt_label, 'pred_class':pred_label, 'score_pred':score_pred, 'score_gt': score_gt, 'filenames': filenames})
    results.to_csv("eval_results/confi_score_{}_{}.csv".format(model_type, data_type), mode='w+')

    