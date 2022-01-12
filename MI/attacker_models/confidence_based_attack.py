from __future__ import print_function
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import cifar10
from keras.datasets import cifar100
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score
from matplotlib import pyplot as plt
from matplotlib import rcParams
from Utilities import APV, APV2D, FAR
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from attacker_models.conf_based_utils import compute_conf_train_average, compute_conf_test_average, per_class_labelling, prepare_dataset

rcParams.update({'font.size': 16})
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)

balanceness_ration = 5
apply_sampling_to_test_as_well = False
Show_MI_attack = True




def confidence_attack(dataset, attack_classifier, sampling, attacker_knowledge, save_conf_histogram, report_separated_performance, num_classes, num_targeted_classes, model_name, verbose):
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        
        
    # convert class vectors to binary class matrix
    
    y_train = keras.utils.to_categorical(y_train, num_classes) # converts class vectors(integer) to binary matrix
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    X_train = X_train.astype('float32')
    y_test = y_test.astype('float32')
    
    if dataset == 'mninst':
        X_train /= 255
        X_test /= 255
        
    model = tf.keras.models.load_model(model_name)
    
    train_stat = model.evaluate(X_train, y_train, verbose=0) # returns test-loss and test-accuracy
    test_stat = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Training Data Stats (test-loss, accuracy): {train_stat}")
    print(f"Testing Data Stats (test-loss, accuracy): {test_stat}")
    
    
    conf_train = model.predict(X_train)
    conf_test = model.predict(X_test)
    
    print(conf_train, conf_test)
    
    labelsTrained = np.argmax(conf_train, axis=1)
    labelsTest = np.argmax(conf_test, axis=1)
    
    print(labelsTrained, labelsTest)
    
    labels_train = np.argmax(y_train, axis=1)
    labels_test = np.argmax(y_test, axis=1)
    
    print(labels_train.shape, labels_test.shape)
    
   
    # compute average for train set.
    (correctlyClassifiedIndex_Train, incorrectlyClassifiedIndex_Train) = compute_conf_train_average(labelsTrained, labels_train, conf_train)
    
    # compute average for test set
    (correctlyClassifiedIndex_Test, incorrectlyClassifiedIndex_Test) = compute_conf_test_average(labelsTest, labels_test, conf_test )
    
    # per class labelling
    per_class_labelling( num_classes, num_targeted_classes)
    
    # prepare_dataset_attacker()
    prepare_dataset(correctlyClassifiedIndex_Train, incorrectlyClassifiedIndex_Train, correctlyClassifiedIndex_Test, incorrectlyClassifiedIndex_Test, num_targeted_classes, conf_train, conf_test,labelsTrained, labels_train, labelsTest, labels_test)
    
    