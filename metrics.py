'''
This script contains functions to define the metrics used in CAM faithfulness evaluation
'''

import os
import numpy as np

epsilon = 1e-7

def distL2(M1, M2):
    # Distance based on L2 distance.
    return np.sum(np.square(M1 - M2), axis=(1,2))

def distPCC(M1, M2):
    # Distance based on Pearson Correlation Coefficient. Details in TPAMI2018 paper 
    # "What do different evaluation metrics tell us about saliency models?"
	m_M1, m_M2 = np.mean(M1, axis=(1,2), keepdims=True), np.mean(M2, axis=(1,2), keepdims=True)
	r_num = np.sum((M1 - m_M1) * (M2 - m_M2), axis=(1,2))
	r_den = np.sqrt(np.sum(np.square(M1 - m_M1), axis=(1,2)) * np.sum(np.square(M2 - m_M2), axis=(1,2)))
	ratio = r_num / (r_den + epsilon)
	ratio = np.clip(ratio, -1.0, 1.0)
	return 1 - ratio

def toProbDistro(M):
    # Convert the matrix to a 2D distribution (i.e. normalization).
	return M / np.sum(M, axis=(1,2), keepdims=True)

def distKL(P1, P2):
    # Distance based on KL divergence
    return np.sum(P1 * np.log(epsilon + P1 / (P2 + epsilon)), axis=(1,2))

def distJS(M1, M2):
    # Distance based on JS divergence
    P1 = toProbDistro(M1)
    P2 = toProbDistro(M2)
    Q = (P1 + P2)/ 2
    return (distKL(P1, Q) + distKL(P2, Q)) / 2

def distSIM(M1, M2):
    # Distance based on Similarity. Details in TPAMI2018 paper 
    # "What do different evaluation metrics tell us about saliency models?"
	P1 = toProbDistro(M1)
	P2 = toProbDistro(M2)
	similarity = 1.0 / 2 * np.sum(P1 + P2 - np.abs(P2 - P1), axis=(1,2))
	return 1 - similarity

def get_r_square(y_true, y_pred):
    # Coefficient of determination (R^2) for regression
    SS_res =  np.sum(np.square(y_true - y_pred)) 
    SS_tot = np.sum(np.square(y_true - np.mean(y_true))) 
    return (1 - SS_res/(SS_tot + epsilon))