import emoji
import io
import numpy
import re
from autocorrect import spell

trainDataPath = "data/test.txt"

emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}

def preprocessData(dataFilePath, mode):
    indices = []
    conversations = []
    labels = []

    vocabulary = []
    with io.open(dataFilePath, encoding = 'utf-8') as finput:
        finput.readline()
        for line in finput:
            repeatedChars = ['.', '?', '!', ',']
            repeatedChars.extend([char for char in line if char in emoji.UNICODE_EMOJI])
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '    
                line = cSpace.join(lineSplit)
            
            line = line.strip().split('\t')
            if mode == "train":
                label = emotion2label[line[4]]
                labels.append(label)
            
            conv = ' <eos> '.join(line[1:4]).lower()
            conv = re.sub('\d+', ' ', conv)
            conv = re.sub('(\w)\\1\\1+', '\\1', conv)
            conv = re.sub('\s+', ' ', conv)

            split = conv.split()
            for i in range(len(split)):
                if split[i] not in vocabulary:
                    correction = spell(split[i])
                    vocabulary.append(correction)
                    split[i] = correction
            
            conv = ' '.join(split)

            index = int(line[0])
            indices.append(index)
            conversations.append(conv.lower())
            
            if index % 1000 == 0:
                print(index)
    
    numpy.savez('test.npz', indices = indices, conversations = conversations, labels = labels)

    print(indices[0])
    print(conversations[0])
    print(labels[0])

preprocessData(trainDataPath, mode = 'train')