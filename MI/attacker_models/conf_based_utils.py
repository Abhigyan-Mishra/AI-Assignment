
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score
from Utilities import APV, APV2D, FAR
from tensorflow.keras.layers import Dense



def compute_conf_train_average(labelsTrained, labels_train, conf_train):
    print("\n ******************** TRAIN SET ********************")
    tempIndexer = np.arange(conf_train.shape[0]) # [0 - 59999]
    print("TempIndexer", tempIndexer)
    tempConfArray = conf_train[tempIndexer, labels_train]  # confidence value for every single data point, the probability that each data point will be properly classified.
    confTrain = np.average(tempConfArray)  # average them out so that it give me proper (normalized value), not bias towards bigger value.
    confTrainSTD = np.std(tempConfArray) # find the standard deviation

    print(f"tempIndexer: {tempIndexer} \
        \ntempConfArray: {tempConfArray} \nconfTrain: {confTrain}  \
            \nconfTrainSTD: {confTrainSTD}")

    correctlyClassifiedIndex_Train = labelsTrained == labels_train   # this is for thos that are correctly classified
    correctConfArray = conf_train[tempIndexer[correctlyClassifiedIndex_Train],
                                  labels_train[correctlyClassifiedIndex_Train]]   # confidence values for correctly classified datapoints.
    correctConfTrain = np.average(correctConfArray)   # find average
    correctConfTrain_STD = np.std(correctConfArray) # find standard deviation

    print(f"correctly classified Index: {correctlyClassifiedIndex_Train} \
        \ncorrect confidence Array: {correctConfArray} \
           \ncorrect confidence trained: {correctConfTrain} \
             \ncorrect confidence trained STD: {correctConfTrain_STD}")

    incorrectlyClassifiedIndex_Train = labelsTrained != labels_train  # for missclassified
    incorrectConfArray = conf_train[tempIndexer[incorrectlyClassifiedIndex_Train],
                                    labelsTrained[incorrectlyClassifiedIndex_Train]] # confidence values of missclassification - how confidence that it will be missclassified.
    incorrectConfTrain = np.average(incorrectConfArray)
    incorrectConfTrain_STD = np.std(incorrectConfArray)


    print(f"incorrectly classified Index: {incorrectlyClassifiedIndex_Train} \
        \n incorrect confidence Array: {incorrectConfArray} \
           \n incorrect confidence trained: {incorrectConfTrain} \
             \n   incorrect confidence trained STD: {incorrectConfTrain_STD}")

    return (confTrain, confTrainSTD, correctlyClassifiedIndex_Train, incorrectlyClassifiedIndex_Train)


def compute_conf_test_average(labelsTest, labels_test, conf_test):
    print("\n ******************** TEST SET ********************")
    tempIndexer = np.arange(conf_test.shape[0])
    confArray = conf_test[tempIndexer, labels_test]
    confTest = np.average(confArray)
    confTest_STD = np.std(confArray)

    correctlyClassifiedIndex_Test = labelsTest == labels_test
    correctConfArray = conf_test[tempIndexer[correctlyClassifiedIndex_Test],
                                 labels_test[correctlyClassifiedIndex_Test]]
    correctConfTest = np.average(correctConfArray)
    correctConfTest_STD = np.std(correctConfArray)

    print("************* TRAIN ***************")
    print(f"correctly classified Index: {correctlyClassifiedIndex_Test} \
        \n correct confidence Array: {correctConfArray} \
           \n correct confidence trained: {correctConfTest} \
             \n   correct confidence trained STD: {correctConfTest_STD}")

    print("************* TEST ***************")
    incorrectlyClassifiedIndex_Test = labelsTest != labels_test
    incorrectConfArray = conf_test[tempIndexer[incorrectlyClassifiedIndex_Test],
                                   labelsTest[incorrectlyClassifiedIndex_Test]]
    incorrectConfTest = np.average(incorrectConfArray)
    incorrectConfArray_STD = np.std(incorrectConfArray)

    print(f"correctly classified Index: {correctlyClassifiedIndex_Test} \
    \n correct confidence Array: {correctConfArray} \
        \n correct confidence trained: {correctConfTest} \
            \n   correct confidence trained STD: {correctConfTest_STD}")

    return (confTest, confTest_STD, correctlyClassifiedIndex_Test, incorrectlyClassifiedIndex_Test)


def getBalancedAccuracy(numTargetedClasses):
    print("\n************Accuracy***************")
    balancedAccuracy = np.zeros(numTargetedClasses) - 1
    correctlyLabeledBalancedAccuracy = np.zeros(numTargetedClasses) - 1
    incorretlyLabeledBalancedAccuracy = np.zeros(numTargetedClasses) - 1

    return (balancedAccuracy, correctlyLabeledBalancedAccuracy, incorretlyLabeledBalancedAccuracy)


def getAccuracy(numTargetedClasses):
    accuracy = np.zeros(numTargetedClasses) - 1
    return (accuracy, accuracy, accuracy)


def getFAR(numTargetedClasses):
    far = np.zeros(numTargetedClasses) - 1
    return(far, far, far)


def getPrecision(numTargetedClasses):
    precision = np.zeros((numTargetedClasses, 2)) - 1
    return (precision, precision, precision)


def getRecall(numTargetedClasses):
    recall = np.zeros((numTargetedClasses, 2)) - 1
    return (recall, recall, recall)


def getF1Score(numTargetedClasses):
    f1Score = np.zeros((numTargetedClasses, 2)) - 1
    return (f1Score, f1Score, f1Score)


'''
@params: numTargetedClasses
$$ placeholder for performance measures to store.
'''


def plot_graph():
    
    print("histogram")
    # t = classYesX[correctlyLabeledYesX]
    # t2 = classNoX[correctlyLabeledNoX]

    # t = np.average(t, axis=0)
    # t2 = np.average(t2, axis=0)

    # plt.style.use('seaborn-deep')
    # plt.plot(np.arange(numClasses), t, 'bx', label="Train samples")
    # plt.plot(np.arange(numClasses), t2, 'go', label="Test samples")

    # plt.legend()
    # plt.xlabel("Class Number")
    # plt.ylabel("Average Confidence")
    # plt.savefig("figures/conf_histogram/" + dataset + '/correct-' + str(j) + '.png', bbox_inches='tight')
    # plt.close()

    # t = classYesX[incorrectlyLabeledYesX]
    # t2 = classNoX[incorrectlyLabeledNoX]
    # t = np.average(t, axis=0)
    # t2 = np.average(t2, axis=0)
    # plt.style.use('seaborn-deep')
    # plt.plot(np.arange(numClasses), t, 'bx', label="Train Samples")
    # plt.plot(np.arange(numClasses), t2, 'go', label="Test Samples")
    # plt.legend()
    # plt.xlabel('Class Number')
    # plt.ylabel('Average Confidence')
    # plt.savefig('figures/conf_histogram/' + dataset + '/misclassified-' + str(j) + '.png', bbox_inches='tight')
    # plt.close()

    # t = classYesX[correctlyLabeledYesX]
    # t2 = classNoX[correctlyLabeledNoX]

    # bins = np.arange(101) / 100
    # plt.style.use('seaborn-deep')
    # n, bins, patches = plt.hist([t[:, j], t2[:, j]], bins,  alpha=1, label=["Train Samples", "Test Samples"])

    # plt.xlabel('Model Confidence')
    # plt.ylabel('Probability (%)')
    # plt.legend(loc="upper left")
    # plt.savefig('figures/conf_histogram/' + dataset + '/' + str(j) + '.png', bbox_inches='tight')
    # plt.close()



def to_store_p_measures(numClasses, numTargetedClasses):

    (balancedAccuracy, correctlyLabeledBalancedAccuracy,
     incorrectlyLabeledBalancedAccuracy) = getBalancedAccuracy(numTargetedClasses)
    (accuracy, correctlyLabeledAccuracy,
     incorrectlyLabeledAccuracy) = getAccuracy(numTargetedClasses)
    (far, correctlyLabeledFar, incorrectlyLabeledFar) = getFAR(numTargetedClasses)

    (precision, correctlyLabeledPrecision,
     incorrectlyLabeledPrecision) = getPrecision(numTargetedClasses)

    (recall, correctlyLabeledRecall,
     incorrectlyLabeledRecall) = getRecall(numTargetedClasses)

    (f1score, correctlyLabeledF1score,
     incorrectlyLabeledF1score) = getF1Score(numTargetedClasses)
    return (
        balancedAccuracy,
        correctlyLabeledBalancedAccuracy,
        incorrectlyLabeledBalancedAccuracy,
        accuracy,
        correctlyLabeledAccuracy,
        incorrectlyLabeledAccuracy,
        far, correctlyLabeledFar, incorrectlyLabeledFar,
        precision, correctlyLabeledPrecision, incorrectlyLabeledPrecision,
        recall, correctlyLabeledRecall, incorrectlyLabeledRecall,
        f1score, correctlyLabeledF1score, incorrectlyLabeledF1score
    )


def attack_classwise(j, dataset, correctlyClassifiedIndex_Train, incorrectlyClassifiedIndex_Train, correctlyClassifiedIndex_Test, incorrectlyClassifiedIndex_Test, numClasses, numTargetedClasses, conf_train, conf_test, labelsTrained, labels_train, labelsTest, labels_test, attacker_knowledge, SHOW_ATTACK, attack_classifier, save_conf_histogram=True):
    print("XXXXXXXXXXXXXXXXXXXXXXX")


    classYesX = conf_train[tuple([labels_train == j])]  # highest at where it matches [9.9997485e-01 7.6881284e-09 7.7287825e-07 2.1447873e-07 1.6986093e-07 1.9313011e-06 5.6364697e-06 3.9415945e-06 6.9429079e-06 5.5773412e-06]

    classNoX = conf_test[tuple([labels_test == j])]
    

    #check if there is enough sample

    if classYesX.shape[0] < 15 or classNoX.shape[0] < 15: 
        print(
            f"Class {str(j)} doesn't have enough sample for training for attack")


    # find the exact classified value 
    correctlyLabeledYesX = correctlyClassifiedIndex_Train[tuple(
        [labels_train == j])]

    
    correctlyLabeledNoX = correctlyClassifiedIndex_Test[tuple(
        [labels_test == j])]

    incorrectlyLabeledYesX = incorrectlyClassifiedIndex_Train[tuple(
        [labels_train == j])]
    incorrectlyLabeledNoX = incorrectlyClassifiedIndex_Test[tuple(
        [labels_test == j])]

    # plot_graph()

    # multiply with what have found out and already known.
    
    classYesSize = int(classYesX.shape[0] * attacker_knowledge)
    classYesXTrain = classYesX[:classYesSize]
    classYesYTrain = np.ones(classYesXTrain.shape[0])

    classYesXTest = classYesX[classYesSize:]
    classYesYTest = np.ones(classYesXTest.shape[0])
    correctlyLabeledYesX = correctlyLabeledYesX[classYesSize:]
    incorrectlyLabeledYesX = incorrectlyLabeledYesX[classYesSize:]

    classNoSize = int(classNoX.shape[0] * attacker_knowledge)
    classNoXTrain = classNoX[:classNoSize]
    classNoYTrain = np.zeros(classNoXTrain.shape[0])
    classNoXTest = classNoX[classNoSize:]
    classNoYTest = np.zeros(classNoXTest.shape[0])
    correctlyLabeledNoX = correctlyLabeledNoX[classNoSize:]
    incorrectlyLabeledNoX = incorrectlyLabeledNoX[classYesSize:]

    Y_size = classYesXTrain.shape[0]
    n_size = classNoXTrain.shape[0]
    print()
    print(f"MI attack on class::  [{j}]")

    X_train = np.concatenate((classYesXTrain, classNoXTrain), axis=0)
    y_train = np.concatenate((classYesYTrain, classNoYTrain), axis=0)
    X_test = np.concatenate((classYesXTest, classNoXTest), axis=0)
    y_test = np.concatenate((classYesYTest, classNoYTest), axis=0)

    correctlyLabeledIndices = np.concatenate(
        (correctlyLabeledYesX, correctlyLabeledNoX), axis=0)
    incorrectlyLabeledIndices = np.concatenate(
        (incorrectlyLabeledYesX, incorrectlyLabeledNoX), axis=0)

    if SHOW_ATTACK:
        if attack_classifier == "NN":
            ATTACK_MODEL = Sequential()
            ATTACK_MODEL.add(
                Dense(128, input_dim=numClasses, activation="relu"))
            ATTACK_MODEL.add(Dense(64, activation="relu"))
            ATTACK_MODEL.add(Dense(1, activation="sigmoid"))

            ATTACK_MODEL.compile(
                loss='binary_crossentropy', optimizer="adam", metrics=['acc'])
            ATTACK_MODEL.fit(X_train, y_train, validation_data=(
                X_test, y_test), epochs=30, batch_size=32, verbose=False, shuffle=True)

            y_pred = ATTACK_MODEL.predict(X_test)
            predictions = np.where(y_pred > 0.8, 1,0)            
            (
                balancedAccuracy,
                correctlyLabeledBalancedAccuracy,
                incorrectlyLabeledBalancedAccuracy,
                accuracy,
                correctlyLabeledAccuracy,
                incorrectlyLabeledAccuracy,
                far, correctlyLabeledFar, incorrectlyLabeledFar,
                precision, correctlyLabeledPrecision, incorrectlyLabeledPrecision,
                recall, correctlyLabeledRecall, incorrectlyLabeledRecall,
                f1score, correctlyLabeledF1, incorrectlyLabeledF1
            ) = to_store_p_measures(numClasses, numTargetedClasses)



            balancedAccuracy[j] = balanced_accuracy_score(y_test, predictions )
            accuracy[j] = accuracy_score(y_test, predictions)
            far[j] = FAR(y_test, predictions)
            precision[j] = precision_score(y_test, predictions, average="weighted", labels=np.unique(predictions))
            recall[j] = recall_score(y_test, predictions)
            f1score[j] = f1_score(y_test, predictions)

            mi, mi_STD = APV(balancedAccuracy)
            c_mi, c_mi_STD = APV(correctlyLabeledBalancedAccuracy)
            in_mi_attack, in_mi_attack_std = APV(
                incorrectlyLabeledBalancedAccuracy)

            mi_acc, mi_accSTD = APV(accuracy)
            c_mi_acc, c_mi_accSTD = APV(correctlyLabeledAccuracy)
            in_mi_acc, in_mi_accSTD = APV(incorrectlyLabeledAccuracy)

            mi_far, mi_farSTD = APV(far)
            c_mi_far, c_mi_farSTD = APV(correctlyLabeledFar)
            in_mi_far, in_mi_farSTD = APV(incorrectlyLabeledFar)

            mi_prec, mi_precSTD = APV2D(precision)
            c_mi_prec, c_mi_precSTD = APV2D(correctlyLabeledPrecision)
            in_mi_prec, in_mi_precSTD = APV2D(incorrectlyLabeledPrecision)

            mi_rcal, mi_rcalSTD = APV2D(recall)
            c_mi_rcal, c_mi_rcalSTD = APV2D(correctlyLabeledRecall)
            in_mi_rcal, in_mi_rcalSTD = APV2D(incorrectlyLabeledRecall)

            mi_f1, mi_f1STD = APV2D(f1score)
            c_mi_f1, c_mi_f1STD = APV2D(correctlyLabeledF1)
            in_mi_f1, in_mi_f1STD = APV2D(incorrectlyLabeledF1)

            return (mi,
                    mi_STD,
                    c_mi, c_mi_STD, in_mi_attack,
                    in_mi_attack_std, mi_acc, mi_accSTD,
                    c_mi_acc, c_mi_accSTD, in_mi_acc,
                    in_mi_accSTD,
                    mi_far, mi_farSTD,
                    c_mi_far, c_mi_farSTD,
                    in_mi_far, in_mi_farSTD,
                    mi_prec, mi_precSTD,
                    c_mi_prec, c_mi_precSTD,
                    in_mi_prec, in_mi_precSTD,

                    mi_rcal, mi_rcalSTD,
                    c_mi_rcal, c_mi_rcalSTD,
                    in_mi_rcal, in_mi_rcalSTD,
                    mi_f1, mi_f1STD,
                    c_mi_f1, c_mi_f1STD,
                    in_mi_f1, in_mi_f1STD
                    )
