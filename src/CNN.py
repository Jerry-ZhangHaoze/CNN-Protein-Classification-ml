#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 03:57:01 2026

@author: hamejerry
"""

# coding: utf-8
# ============ Import ============
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adamax, Nadam, Adagrad, Adadelta
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasClassifier
import random


# ============ LOAD DATA ============
# The dataset is loaded as strings because both protein sequences and class labels are text-based.
dataset = np.loadtxt('trainingdata.csv', skiprows=1, delimiter=",", dtype='str')

# Quick check the column content and sequence length
print(dataset[:, 3])
print(len(dataset[1, 2]))

# ============ ENCODE FUNCTION ============
'''
convert a protein sequence into an encoded matrix.
each amino acid is represented as a 21-dimensional vector: 20 standard amoino acids plus '-' for padding/gaps.
'''
def encode(s_code):
    translate = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10,
                 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20}
    ret = []
    for letter in s_code:
        app = [0] * 21
        app[translate[letter]] = 1
        ret.append(app)
    return np.asarray(ret)

# ============ TOKENIZER ============
# Tokenizer converts text labels into numerical class vectors for multi-class classification.
t = Tokenizer()
t.fit_on_texts(dataset[:, 4])

# Print label vocabulary information for verification
print(t.word_counts)
print(t.document_count)
print(t.word_index)

# Example encoded label vector
encoded_docs = t.texts_to_matrix(dataset[:, 4], mode='count')
print(encoded_docs[0])

# Number of output classes used by the final softmax layer
num_classes = len(t.word_index) + 1

# ============ DATASET 1: Use column 3 sequences to Predict column 4 labels ============
# Split the data into training and testing sets
# random_state makes the split reproducible; stratify preserves label distribution.
train_x, test_x, train_y, test_y = train_test_split(
    dataset[:, 3], 
    dataset[:, 4], 
    test_size=0.15,
    random_state=42,
    stratify=dataset[:, 4])

# Convert protein sequences and labels into numerical matrices.
train_x = np.asarray([encode(a) for a in train_x])
test_x = np.asarray([encode(a) for a in test_x])
train_y = t.texts_to_matrix(train_y, mode='count')
test_y = t.texts_to_matrix(test_y, mode='count')

# Confirm input dimensions for the CNN model
print(np.shape(train_x[0]))
print(np.shape(train_x))

# ============ CNN model for DATASET 1 ============
# Conv1D captures local sequence patterns in the protein representation.
bestmodel = Sequential()
bestmodel.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(10, 21)))
bestmodel.add(Flatten())
bestmodel.add(Dense(num_classes, activation='softmax'))

# Optimzer along with the specific learning rate
optimizer = RMSprop(learning_rate=0.001)

bestmodel.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=optimizer,
    metrics=['accuracy']
    )

# ============ CNN model factors and result for DATASET 1 ============
bestmodel.fit(train_x, train_y, 
              batch_size=128, 
              epochs=150, 
              verbose=1, 
              validation_data=(test_x, test_y))

score = bestmodel.evaluate(test_x, test_y, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Convert redicted probabilities and labels into class indices
y_pred = np.argmax(bestmodel.predict(test_x), axis=1)
y_true = np.argmax(test_y, axis=1)

print(classification_report(y_true, y_pred))

# ============ DATASET 2: Use column 2 sequences to Predict column 4 labels ============
# Column 2 uses longer protein sequence information
train_x1, test_x1, train_y1, test_y1 = train_test_split(
    dataset[:, 2],
    dataset[:, 4],
    test_size=0.15,
    random_state=42,
    stratify=dataset[:, 4])

# Convert the second sequence representation and labels into matrices
train_x1 = np.asarray([encode(a) for a in train_x1])
test_x1 = np.asarray([encode(a) for a in test_x1])
train_y1 = t.texts_to_matrix(train_y1, mode='count')
test_y1 = t.texts_to_matrix(test_y1, mode='count')

bestmodel1 = Sequential()
bestmodel1.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(34, 21)))
bestmodel1.add(Flatten())
bestmodel1.add(Dense(num_classes, activation='softmax'))

optimizer1 = RMSprop(learning_rate=0.001)

bestmodel1.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=optimizer1,
    metrics=['accuracy']
)

# ============ CNN model for DATASET 2 ============
bestmodel1.fit(
    train_x1,
    train_y1,
    batch_size=128,
    epochs=150,
    verbose=1,
    validation_data=(test_x1, test_y1)
)

score1 = bestmodel1.evaluate(test_x1, test_y1, verbose=1)
print('Dataset 2 Test loss:', score1[0])
print('Dataset 2 Test accuracy:', score1[1])

# Convert model outputs into class labels for the classification
y_pred1 = np.argmax(bestmodel1.predict(test_x1), axis=1)
y_true1 = np.argmax(test_y1, axis=1)

print(classification_report(y_true1, y_pred1))



