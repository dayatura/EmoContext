import emoji
import io
import numpy
import re
import string
from autocorrect import spell
from gensim import models
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing import sequence

trainDataPath = 'train.txt'
testDataPath = 'test.txt'
word2vecPath = 'bin/word2vec.bin'
emoji2vecPath = 'bin/emoji2vec.bin'
classifyModelPath = 'model/classify.json'
classifyWeightPath = 'model/classify.h5'
trainVectorPath = 'train.npz'
testVectorPath = 'test.npz'
nominal2numeric = {'others' : 0, 'happy' : 1, 'sad' : 2, 'angry' : 3}

def load_data(path):
    with io.open(path, encoding = 'utf-8') as finput:
        dataX = []
        dataY = []
        finput.readline()
        for line in finput:
            lineSplit = line.strip().split('\t')
            dataX.append(' <eos> '.join(lineSplit[1:4]).lower())
            dataY.append(lineSplit[4])
    return dataX, dataY

def preprocess(dataX, dataY, w2v, e2v, maxCount = 0):
    count = [0, 0, 0, 0]
    oov = []
    corrected = {}
    X = []
    Y = []

    for i in range(len(dataX)):
        if count[nominal2numeric[dataY[i]]] < maxCount or maxCount == 0:
            dataX[i] = dataX[i] + ' <eoc>'
            trimmedChar = ['.', '?', '!', ',']
            trimmedChar.extend([char for char in dataX[i] if char in emoji.UNICODE_EMOJI])
            
            for char in trimmedChar:
                lineSplit = dataX[i].split(char)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                charSpace = ' ' + char + ' '
                dataX[i] = charSpace.join(lineSplit)
            
            dataX[i] = re.sub('\d+', ' ', dataX[i])
            dataX[i] = re.sub('\s+', ' ', dataX[i])
            dataX[i] = re.sub(r'(\w)\1\1+', r'\1', dataX[i])

            sVector = []
            for word in dataX[i].split():
                if word in corrected:
                    sVector.append(w2v[corrected[word]])
                elif word in oov:
                    sVector.append(numpy.zeros(300))
                elif word in w2v:
                    sVector.append(w2v[word])
                elif word in e2v:
                    sVector.append(e2v[word])
                else:
                    corr = spell(word)
                    if corr in w2v:
                        sVector.append(w2v[corr])
                        corrected[word] = corr
                    else:
                        sVector.append(numpy.zeros(300))
                        oov.append(word)
            dataX[i] = sVector
            dataY[i] = nominal2numeric[dataY[i]]
            X.append(dataX[i])
            Y.append(dataY[i])
            count[dataY[i]] += 1
    
    print('oov : {0}'.format(len(oov)))

    dataX = X
    dataY = Y
    enc = OneHotEncoder(handle_unknown = 'ignore')
    X = numpy.zeros((len(dataX), max([len(i) for i in dataX]), 300))
    for i in range(len(dataX)):
        for j in range(len(dataX[i])):
            X[i][j] = dataX[i][j]
    Y = enc.fit_transform(numpy.array(dataY).reshape(-1, 1)).toarray()

    numpy.savez(trainVectorPath, dataX = X, dataY = Y)

def train():
    data = numpy.load(trainVectorPath)
    X = data['dataX']
    Y = data['dataY']

    model = Sequential()
    model.add(LSTM(128, dropout = 0.2))
    model.add(Dense(4, activation = 'sigmoid'))
    #rmsprop = optimizers.RMSprop(lr = 0.003)
    adam = optimizers.Adam(lr = 0.003)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    model.fit(X, Y, epochs = 10, validation_split = 0.2, batch_size = 32, verbose = 1)
    print(model.summary())

    model_json = model.to_json()
    with open(classifyModelPath, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(classifyWeightPath)

def test():
    data = numpy.load(testVectorPath)
    X = data['dataX']
    Y = data['dataY']
    dataY = [numpy.argmax(i) for i in Y]

    model_json = open(classifyModelPath, 'r')
    model = model_from_json(model_json.read())
    model_json.close()
    model.load_weights(classifyWeightPath)

    prediction_nominal = []
    prediction = model.predict(X)
    for i in range(len(prediction)):
        prediction_nominal.append(numpy.argmax(prediction[i]))
    matrix = confusion_matrix(dataY, prediction_nominal)
    accuracy = accuracy_score(dataY, prediction_nominal)
    score = f1_score(dataY, prediction_nominal, average = 'macro')
    print('confusion matrix :')
    print(matrix)
    print('accuracy : {0}'.format(accuracy))
    print('F1 score : {0}'.format(score))

# print('Loading train data...')
# dataX, dataY = load_data(trainDataPath)
# print('Loading word embedding...')
# w2v = models.KeyedVectors.load_word2vec_format(word2vecPath, binary = True)
# print('Loading emoji embedding...')
# e2v = models.KeyedVectors.load_word2vec_format(emoji2vecPath, binary = True)
# print('Preprocessing...')
# preprocess(dataX, dataY, w2v, e2v, 4000)
print('Training model...')
train()
# print('Loading test data...')
# dataX, dataY = load_data(testDataPath)
# print('Preprocessing...')
# preprocess(dataX, dataY, w2v, e2v, 0)
print('Test model...')
test()