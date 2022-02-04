import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv("datasets/wiscon.csv")
#Full Dataset
feature_set=['ClumpThickness', ' CellSize', ' CellShape', ' MarginalAdhesion',
       ' EpithelialSize', ' BareNuclei', ' BlandChromatin', ' NormalNucleoli',
       ' Mitoses']
target='class'
X_train, X_test, y_train, y_test = train_test_split(df[feature_set], df[target], test_size=0.2, random_state=7) 
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y_test = labelencoder_X_1.fit_transform(y_test)
y_train= labelencoder_X_1.fit_transform(y_train)
# Logistic Regression
lr = LogisticRegression(solver='liblinear', random_state=7)
lr.fit(X_train,y_train)    
print("Accuracy of Logistic Regression(Full Dataset):",accuracy_score(y_test, lr.predict(X_test)))
#Keras NN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 
from keras.layers import Dense, Activation
from keras.layers import Dropout
model = Sequential()
model.add(Dense(9, activation='sigmoid', input_shape=(9,)))
model.add(Dense(27, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(54, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(27, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_logarithmic_error)
model.fit(X_train, y_train, batch_size=30, epochs=2000, verbose=0, validation_data=(X_test, y_test))
loss = model.evaluate(X_test, y_test, verbose=0, batch_size=30)
print("Accuracy of NN(Full Dataset):, {}".format(100 - loss*100))


#Genetic Features
feature_set=['ClumpThickness',' CellShape',
       ' EpithelialSize', ' BareNuclei', ' BlandChromatin' ]
target='class'
X_train, X_test, y_train, y_test = train_test_split(df[feature_set], df[target], test_size=0.2, random_state=7)
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y_test = labelencoder_X_1.fit_transform(y_test)
y_train= labelencoder_X_1.fit_transform(y_train)
# Logistic Regression
lr = LogisticRegression(solver='liblinear', random_state=7)
lr.fit(X_train,y_train)    
print("Accuracy of Logistic Regression(Feature Picked):",accuracy_score(y_test, lr.predict(X_test)))
#Keras NN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 
from keras.layers import Dense, Activation
from keras.layers import Dropout
model = Sequential()
model.add(Dense(9, activation='sigmoid', input_shape=(9,)))
model.add(Dense(27, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(54, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(27, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_logarithmic_error)
model.fit(X_train, y_train, batch_size=30, epochs=2000, verbose=1, validation_data=(X_test, y_test))
loss = model.evaluate(X_test, y_test, verbose=0, batch_size=30)
print("Accuracy of NN(Feature Picked):, {}".format(100 - loss*100))

