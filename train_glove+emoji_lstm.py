############################# load library #################################
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, TimeDistributed, Flatten, Activation, RepeatVector, Permute, Multiply, Lambda
from keras import optimizers
from keras.models import load_model
import keras.backend as K
import json, argparse, os
import re
import io
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

############################# config #################################

trainDataPath = "data/train.npz"
testDataPath = "data/test.npz"
pre_trainedPath = "pre_trained"
embedingName = "glove+emoji.txt"
modelPath = "model/glove+emo2vec_lstm.h5"

NUM_CLASSES = 4                 
MAX_NB_WORDS = 20000                
MAX_SEQUENCE_LENGTH = 100        
EMBEDDING_DIM = 100               
BATCH_SIZE = 200                 
LSTM_DIM = 128                    
DROPOUT = 0.2        
LEARNING_RATE = 0.003
NUM_EPOCHS = 10


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}

############################ helpen function ###########################

def getEmbeddingMatrix(wordIndex):
    embeddingsIndex = {}
    with io.open(os.path.join(pre_trainedPath, embedingName), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector
    
    print('Found %s word vectors.' % len(embeddingsIndex))
    
    oov = 0
    oovWord = []
    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            embeddingMatrix[i] = embeddingVector
        else:
            oovWord += [word]
            oov += 1
    
    print('Found %s unknown word.' % oov)
    
    return embeddingMatrix, oovWord

def buildModel(embeddingMatrix):
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(LSTM(LSTM_DIM, dropout=DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    
    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model

####################### main programm #########################

print("Processing training data...")
# trainIndices, trainTexts, labels = preprocessData(trainDataPath, mode="train")
train = np.load(trainDataPath)
trainIndices = train['indices']
trainTexts = train['conversations']
labels = train['labels']

print("Processing test data...")
# testIndices, testTexts , testLabels = preprocessData(testDataPath, mode="train")
test = np.load(testDataPath)
testIndices = test['indices']
testTexts = test['conversations']
testLabels = test['labels']

print("Extracting tokens...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(trainTexts)
trainSequences = tokenizer.texts_to_sequences(trainTexts)
testSequences = tokenizer.texts_to_sequences(testTexts)

wordIndex = tokenizer.word_index
print("Found %s unique tokens." % len(wordIndex))

print("Populating embedding matrix...")
embeddingMatrix, oovWord = getEmbeddingMatrix(wordIndex)
print("Shape of embeding matrix: ", embeddingMatrix.shape)

data = pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
print("Shape of training data tensor: ", data.shape)
print("Shape of label tensor: ", labels.shape)

# Randomize data
np.random.shuffle(trainIndices)
data = data[trainIndices]
labels = labels[trainIndices]

print("Build and train model...")
model = buildModel(embeddingMatrix)
history = model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
model.save(modelPath)
model = load_model(modelPath)


print("Test model...")
testData = pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)
predictions = model.predict(testData, batch_size=BATCH_SIZE)
predictions = predictions.argmax(axis=1)

# test metrics
matrix = confusion_matrix(testLabels, predictions)
accuracy = accuracy_score(testLabels, predictions)
score = f1_score(testLabels, predictions, average = 'macro')
print('Confusion matrix :')
print(matrix)
print('Accuracy : {0}'.format(accuracy))
print('F1 score : {0}'.format(score))