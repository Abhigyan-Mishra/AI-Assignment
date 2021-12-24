# experiment with different setting and save in different epochs(epoch: one pass over the entire dataset)
from __future__ import print_function
import argparse
import json
import os
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, AveragePooling2D
from keras.models import model_from_config



"""
batch_size: a hyperparameter that defines the number of samples to work through before updating the internal model parameters. 
epochs: passes over the entire datasets.
learning_rate
"""

parser = argparse.ArgumentParser(description='Train shadow model.')
parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=["mnist"]) # can provide multiple choices to choose from.
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of passes required')
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='learning rate.')
args = parser.parse_args()


def create_train_model_dir(dataset):
    model_dir = os.path.join(os.getcwd(), 'shadow_models/' + dataset)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        
    model_name = dataset + ".h5"
    return (model_dir, model_name)
    

def train_shadow_model():
    data = args.dataset
    batchSize = args.batch_size
    numberOfEpochs = args.epochs
    learningRate = args.learning_rate
    
    (model_dir, model_name) = create_train_model_dir(data)
    
    if data == 'mnist':
        num_classes = 10 # tweek required to get better score.
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        

    # convert class vector into binary class matrices.
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    X_train = X_train.astype('float32')
    X_test =  X_test.astype('float32') 
    X_train /= 255
    X_test /= 255
    
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'{X_train.shape[0]} train samples')
    print(f'{X_test.shape[0]} test samples')


    if data == "mnist":
        model = keras.Sequential()
        model.add(Conv2D(filters=5, kernel_size=(3,3), activation='relu', input_shape=X_train.shape[1:]))
        model.add(AveragePooling2D())
        model.add(Conv2D(filters=5, kernel_size=(3,3), activation='relu'))
        model.add(AveragePooling2D())
        
        model.add(Flatten())
        model.add(Dense(units=120, activation='relu'))
        model.add(Dense(units=84, activation='relu'))
        model.add(Dense(units=num_classes, activation='softmax'))
        
        
    optimizer = keras.optimizers.Adam(lr=learningRate, beta_1=0.5, beta_2=0.99, epsilon=1e-08)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=numberOfEpochs, shuffle=True, batch_size=batchSize)

    modelPath = os.path.join(model_dir, model_name)
    model.save(modelPath)
    print(f'Model save at {modelPath}')
    
    scores = model.evaluate(X_train, y_train, verbose=1) 
    print(f'Train Loss: {scores[0]} \t Train Accuracy: {scores[1]}')
    
    scores = model.evaluate(X_test, y_test, verbose=1)
    print(f'Train Loss: {scores[0]} \t Train Accuracy: {scores[1]}')
    
    
       

if __name__ == '__main__':
    train_shadow_model()
