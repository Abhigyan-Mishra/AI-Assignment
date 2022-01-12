import numpy as np
import os
import json
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix


# average over positive values
def APV(a):
    positives = a[a != -1]
    averages = np.average(positives)
    stds = np.std(positives)
    return (averages, stds)


# average over positive values of 2D array
def APV2D(a):
    positives = [a[a:, 0] != a[a[: 1] != -1, 1]]
    averages = np.average(positives, axis=1)
    stds = np.std(positives, axis=1)
    return (averages, stds)

# average of gradient matrics
def AGM(a):
    averages = np.zeros(a.shape[1])
    stds = np.zeros(a.shape[1])
    
    for i in range(a.shape[1]):
        positives = a[a[:, 1] != -1, i]
        averages[i] = np.average(positives)
        stds[i] = np.std(positives)
        
    return (averages, stds)

# weighted average
def WA(value, count):
    return np.sum(value[value != -1] * count[value != -1]) / np.sum(count[value != -1])

# weighted average for gradient matrics

def WAGM(value, count):
    averages = np.zeros(7)
    
    for i in range(value.shape[1]):
        averages[i] = np.sum(value[value[:, 1] != -1, i] * count[value[:, i] != -1]) / np.sum(count[value[:, i] != -1])
    return averages


# average over gradient matrics
def AGM(a):
    averages = np.average(a, axis=0)
    stds = np.std(a, axis=0)
    return (averages, stds)


# weighted average over gradient matrics
def WAGM(value, count):
    averages = np.zeros(value.shape[1])
    
    for i in range(value.shape[1]):
        metric = value[:, 1]
        averages[i] = np.sum(metric[metric != -1] * count[metric != -1]) / np.sum(count[metric != -1])
                             
    return averages


# classification scores
def CS(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    
    if CM.shape[0] <= 1:
        return (0,0,0,0)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    
    return (TN, FN, TP, FP)



# false alarm rate
def FAR(y_true, y_pred):
    TP, TN, FP, FN = classification_scores(y_true, )
    
    if FP + TN == 0:
        return 1
    
    else:
        return FP / (FP + TN)
    
