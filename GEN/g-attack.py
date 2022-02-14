import argparse
import os
import pandas as pd
import numpy as np
from q import q_f
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,  recall_score, precision_score
from sklearn.linear_model import LogisticRegression

NUM_CLASSES = 1

def fit_model(df):
    feature_set=['ClumpThickness', ' CellSize', ' CellShape', ' MarginalAdhesion',
       ' EpithelialSize', ' BareNuclei', ' BlandChromatin', ' NormalNucleoli',
       ' Mitoses']
    target='class'
    y = df["class"]
    X = df.drop(["class"], axis=1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7) 
    labelencoder_X_1 = LabelEncoder()
    y_test = labelencoder_X_1.fit_transform(y_test)
    y_train= labelencoder_X_1.fit_transform(y_train)
    lr = LogisticRegression(solver='liblinear', random_state=7)
    lr.fit(X_train,y_train)    
    y_pred = lr.predict(X_test)
    return  precision_score(y_test, y_pred), recall_score(y_test, y_pred), accuracy_score(y_test, y_pred)

def attack():
    print("Attacking")
    data = pd.read_csv("./datasets/wiscon.csv", index_col=False)
    qubo = pd.read_csv("./datasets/wiscon_qubo.csv")

    measures =  q_f(data, qubo)
    selected, notselected = [], []
    for i in range (0, 171):
       if  measures["class_probabilities"][i][0] == 1.0 :
            selected.append(data.iloc[i].values)
       else:
            notselected.append(data.iloc[i].values)

    selected = pd.DataFrame(selected)
    notselected = pd.DataFrame(notselected)
    selected.columns = ["ClumpThickness", "CellSize","CellShape", "MarginalAdhesion", "EpithelialSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "class" ]
    notselected.columns = ["ClumpThickness", "CellSize","CellShape", "MarginalAdhesion", "EpithelialSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "class" ]
    s_precision, s_recall, s_accuracy = fit_model(selected)
    n_precision, n_recall, n_accuracy = fit_model(notselected)
    print("Selected \n")
    print(f"Precision: {s_precision} \nRecall: {s_recall} \nAccuracy: {s_accuracy}")
    print()
    print("Not Selected \n")
    print(f"Precision: {n_precision} \nRecall: {n_recall} \nAccuracy: {n_accuracy}")

if __name__ == "__main__":
    attack()
    

