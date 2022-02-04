from __future__ import print_function
import tensorflow as tf
import keras
from keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score
from matplotlib import pyplot as plt
from matplotlib import rcParams
from Utilities import APV, APV2D, FAR
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from attacker_models.conf_based_utils import compute_conf_train_average, compute_conf_test_average, attack_classwise, to_store_p_measures

rcParams.update({'font.size': 16})
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)

balanceness_ration = 5
apply_sampling_to_test_as_well = False
SHOW_ATTACK = True


def print_scores(train_acc, test_acc, confTrain, confTrainSTD, confTest,
                 confTest_STD, mi, mi_STD, c_mi, c_mi_STD, in_mi, in_mi_STD, mi_acc, mi_accSTD, c_mi_acc, c_mi_accSTD, in_mi_acc, in_mi_accSTD,
                 mi_far, mi_farSTD, c_mi_far, c_mi_farSTD, in_mi_far, in_mi_farSTD, mi_prec, mi_precSTD, c_mi_prec,
                 c_mi_precSTD, in_mi_prec, in_mi_precSTD, mi_rcal, mi_rcalSTD, c_mi_rcal, c_mi_rcalSTD, in_mi_rcal,
                 in_mi_rcalSTD, mi_f1, mi_f1STD, c_mi_f1, c_mi_f1STD, in_mi_f1, in_mi_f1STD):

    print("\nTarget model accuracy [train vs test]:")
    print(f" {str(np.round(train_acc*100, 2))}, {str(np.round(test_acc*100, 2))}")
    print("\nTarget model confidence [average STD]:")
    print(str(np.round(confTrain*100, 2)), str(np.round(confTrainSTD*100, 2)),
          str(np.round(confTest*100, 2)), str(np.round(confTest_STD*100, 2)))

    print("\n\n\nMI Attack bal. accuracy [average standard_deviation]:")
    print(str(np.round(mi*100, 2)),
          str(np.round(mi_STD*100, 2)))

    print(str(np.round(c_mi*100, 2)), str(np.round(c_mi_STD*100, 2)),
          str(np.round(in_mi*100, 2)), str(np.round(in_mi_STD*100, 2)))

    print("\n\n\nMI Attack FAR [average standard_deviation]:")
    print(str(np.round(mi_far*100, 2)),
          str(np.round(mi_farSTD*100, 2)))
    print(str(np.round(c_mi_far*100, 2)), str(np.round(c_mi_farSTD*100, 2)),
          str(np.round(in_mi_far*100, 2)), str(np.round(in_mi_farSTD*100, 2)))

    print("\n\n\nMI Attack unbal. accuracy [average standard_deviation]:")
    print(str(np.round(mi_acc*100, 2)),
          str(np.round(mi_accSTD*100, 2)))
    print(str(np.round(c_mi_acc*100, 2)), str(np.round(c_mi_accSTD*100, 2)),
          str(np.round(in_mi_acc*100, 2)), str(np.round(in_mi_accSTD*100, 2)))

    print(
        "\nMI Attack precision [average(negative_class) average(positive_class)] [standard_deviation(negative) standard_deviation(positive_class)]:")
    print(str(np.round(mi_prec*100, 2)),
          str(np.round(mi_precSTD*100, 2)))
    print(str(np.round(c_mi_prec*100, 2)), str(np.round(c_mi_precSTD*100, 2)),
          str(np.round(in_mi_prec*100, 2)), str(np.round(in_mi_precSTD*100, 2)))

    print(
        "\nMI Attack recall [[average(negative_class) average(positive_class)] [standard_deviation(negative) standard_deviation(positive_class)]:")
    print(str(np.round(mi_rcal*100, 2)),
          str(np.round(mi_rcalSTD*100, 2)))
    print(str(np.round(c_mi_rcal*100, 2)), str(np.round(c_mi_rcalSTD*100, 2)),
          str(np.round(in_mi_rcal*100, 2)), str(np.round(in_mi_rcalSTD*100, 2)))

    print(
        "\nMI Attack f1 [average(negative_class) average(positive_class)] [standard_deviation(negative) standard_deviation(positive_class)]:")
    print(str(np.round(mi_f1*100, 2)),
          str(np.round(mi_f1STD*100, 2)))
    print(str(np.round(c_mi_f1*100, 2)), str(np.round(c_mi_f1STD*100, 2)),
          str(np.round(in_mi_f1*100, 2)), str(np.round(in_mi_f1STD*100, 2)))


def confidence_attack(dataset, attack_classifier, sampling, attacker_knowledge, save_conf_histogram, num_classes, num_targeted_classes, model_name, verbose):
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # convert class vectors to binary class matrix

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    X_train = X_train.astype('float32')
    y_test = y_test.astype('float32')

    if dataset == 'mninst':
        X_train /= 255
        X_test /= 255

    model = tf.keras.models.load_model(model_name)

    # returns test-loss and test-accuracy
    train_stat = model.evaluate(X_train, y_train, verbose=0)
    test_stat = model.evaluate(X_test, y_test, verbose=0)


    # test accuracy and train accuracy 

    train_acc = train_stat[1]
    test_acc = test_stat[1]



# asked the shadow model to predict the test sample

    # returns  the tensor for each datapoint [10 classes - it will return how likely to go to each class]
    conf_train = model.predict(X_train)
    conf_test = model.predict(X_test)



    # the conf train tensor will be classify to the group that has the highest value.
    labelsTrained = np.argmax(conf_train, axis=1) # we have already used one hot encoder, [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] = 5
    labelsTest = np.argmax(conf_test, axis=1)

    print("labels", labelsTrained)
    print("y_train[0]", y_train[0])
    labels_train = np.argmax(y_train, axis=1) # this will give exact value (5)
    labels_test = np.argmax(y_test, axis=1)

    print("labels_train[0]", labels_train[0])

    '''
    @params: labels_train, labelsTrained, conf_train
    @return: correctly classified and misclassified train dataset

    '''

    (confTrain, confTrainSTD, correctlyClassifiedIndex_Train, incorrectlyClassifiedIndex_Train) = compute_conf_train_average(
        labelsTrained, labels_train, conf_train)

    
    (confTest, confTest_STD, correctlyClassifiedIndex_Test, incorrectlyClassifiedIndex_Test) = compute_conf_test_average(
        labelsTest, labels_test, conf_test)

    for j in range(num_classes):
        (mi, mi_STD, c_mi, c_mi_STD, in_mi, in_mi_STD, mi_acc,
         mi_accSTD, c_mi_acc, c_mi_accSTD, in_mi_acc, in_mi_accSTD,
         mi_far, mi_farSTD, c_mi_far, c_mi_farSTD, in_mi_far, in_mi_farSTD,
         mi_prec, mi_precSTD, c_mi_prec, c_mi_precSTD, in_mi_prec, in_mi_precSTD,
         mi_rcal, mi_rcalSTD, c_mi_rcal, c_mi_rcalSTD, in_mi_rcal, in_mi_rcalSTD,
         mi_f1, mi_f1STD, c_mi_f1, c_mi_f1STD, in_mi_f1, in_mi_f1STD)  = attack_classwise(j, dataset, correctlyClassifiedIndex_Train, incorrectlyClassifiedIndex_Train, correctlyClassifiedIndex_Test, incorrectlyClassifiedIndex_Test, num_classes, num_targeted_classes, conf_train, conf_test, labelsTrained, labels_train, labelsTest, labels_test, attacker_knowledge, SHOW_ATTACK, attack_classifier,  save_conf_histogram)

        print_scores(train_acc, test_acc, confTrain, confTrainSTD, confTest, confTest_STD, mi, mi_STD, c_mi, c_mi_STD, in_mi, in_mi_STD, mi_acc,
                     mi_accSTD, c_mi_acc, c_mi_accSTD, in_mi_acc, in_mi_accSTD,
                     mi_far, mi_farSTD, c_mi_far, c_mi_farSTD, in_mi_far, in_mi_farSTD,
                     mi_prec, mi_precSTD, c_mi_prec, c_mi_precSTD, in_mi_prec, in_mi_precSTD,
                     mi_rcal, mi_rcalSTD, c_mi_rcal, c_mi_rcalSTD, in_mi_rcal, in_mi_rcalSTD,
                     mi_f1, mi_f1STD, c_mi_f1, c_mi_f1STD, in_mi_f1, in_mi_f1STD)
