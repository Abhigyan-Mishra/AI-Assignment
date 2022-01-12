
import numpy as np

def compute_conf_train_average(labelsTrained, labels_train, conf_train):
    print("\n ******************** TRAIN SET ********************")
    tempIndexer = np.arange(conf_train.shape[0])
    tempConfArray = conf_train[tempIndexer, labels_train]
    confTrain = np.average(tempConfArray)
    confTrainSTD = np.std(tempConfArray)
    
    print(f"tempIndexer: {tempIndexer} \
        \n tempConfArray: {tempConfArray} \n confTrain: {confTrain}  \
            \n confTrainSTD: {confTrainSTD}")
    
    
    correctlyClassifiedIndex_Train = labelsTrained == labels_train
    correctConfArray = conf_train[tempIndexer[correctlyClassifiedIndex_Train], labels_train[correctlyClassifiedIndex_Train]]
    correctConfTrain = np.average(correctConfArray)
    correctConfTrain_STD = np.std(correctConfArray)
    
    
    print(f"correctly classified Index: {correctlyClassifiedIndex_Train} \
        \n correct confidence Array: {correctConfArray} \
           \n correct confidence trained: {correctConfTrain} \
             \n   correct confidence trained STD: {correctConfTrain_STD}" )
    
    incorrectlyClassifiedIndex_Train = labelsTrained != labels_train
    incorrectConfArray = conf_train[tempIndexer[incorrectlyClassifiedIndex_Train], labelsTrained[incorrectlyClassifiedIndex_Train]]
    incorrectConfTrain = np.average(incorrectConfArray)
    incorrectConfTrain_STD = np.std(incorrectConfArray)
    
    
    print(f"incorrectly classified Index: {incorrectlyClassifiedIndex_Train} \
        \n incorrect confidence Array: {incorrectConfArray} \
           \n incorrect confidence trained: {incorrectConfTrain} \
             \n   incorrect confidence trained STD: {incorrectConfTrain_STD}" )

    return (correctlyClassifiedIndex_Train, incorrectlyClassifiedIndex_Train)


def compute_conf_test_average(labelsTest, labels_test, conf_test):
    print("\n ******************** TEST SET ********************")
    tempIndexer = np.arange(conf_test.shape[0])
    confArray = conf_test[tempIndexer, labels_test]
    confTest = np.average(confArray)
    confTest_STD = np.std(confArray)
    
    correctlyClassifiedIndex_Test = labelsTest == labels_test
    correctConfArray = conf_test[tempIndexer[correctlyClassifiedIndex_Test], labels_test[correctlyClassifiedIndex_Test]]
    correctConfTest = np.average(correctConfArray)
    correctConfTest_STD = np.std(correctConfArray)
    
    print("************* TRAIN ***************")    
    print(f"correctly classified Index: {correctlyClassifiedIndex_Test} \
        \n correct confidence Array: {correctConfArray} \
           \n correct confidence trained: {correctConfTest} \
             \n   correct confidence trained STD: {correctConfTest_STD}" )
    
    
    print("************* TEST ***************")    
    incorrectlyClassifiedIndex_Test = labelsTest != labels_test
    incorrectConfArray = conf_test[tempIndexer[incorrectlyClassifiedIndex_Test], labelsTest[incorrectlyClassifiedIndex_Test]]
    incorrectConfTest = np.average(incorrectConfArray)
    incorrectConfArray_STD = np.std(incorrectConfArray)

    print(f"correctly classified Index: {correctlyClassifiedIndex_Test} \
    \n correct confidence Array: {correctConfArray} \
        \n correct confidence trained: {correctConfTest} \
            \n   correct confidence trained STD: {correctConfTest_STD}" )

    return (correctlyClassifiedIndex_Test, incorrectlyClassifiedIndex_Test)

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
    
    
def per_class_labelling(numClasses, numTargetedClasses):
    
    (balancedAccuracy, correctlyLabaledBalancedAccuracy, incorrectlyLabeledBalancedAccuracy) = getBalancedAccuracy(numTargetedClasses)
    (accuracy, correctlyLabeledAccuracy, incorrectlyLabeledAccuracy) = getAccuracy(numTargetedClasses)
    (far, correctLabeledFar, incorrectlyLabeledFar) = getFAR(numTargetedClasses)
    (precision, correctlyLabeledPrecision, incorrectlyLabeledPrecision) = getPrecision(numTargetedClasses)
    (recall, correctlyLabeledRecall, incorrectlyLabeledRecall) = getRecall(numTargetedClasses)
    (f1score, correctlyLabeledF1score, incorrectlyLabeledF1score) = getF1Score(numTargetedClasses)
    
    
    
def prepare_dataset(correctlyClassifiedIndex_Train, incorrectlyClassifiedIndex_Train, correctlyClassifiedIndex_Test, incorrectlyClassifiedIndex_Test, numTargetedClasses, conf_train, conf_test, labelsTrained, labels_train, labelsTest, labels_test):
    print("XXXXXXXXXXXXXXXXXXXXXXX")
    for j in range(numTargetedClasses):
        classYesX = conf_train[tuple([labels_train == j])]
        classNoX = conf_test[tuple([labels_test == j])]
        
        if classYesX.shape[0] < 15 or classNoX.shape[0] < 15:
            print(f"Class {str(j)} doesn't have enough sample for training for attack")
            continue
        
        
    
