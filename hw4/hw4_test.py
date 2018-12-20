import pandas as pd
import numpy as np
import re
import sys
import jieba

def load_dataX(path):
    with open(path, 'r', encoding='utf-8') as f:
        readin = f.readlines()
        sentences = []
        for s in readin[1:]:
            s = re.sub('^[0-9]+,', '', s)
            s = s[:-2]
            for same_char in re.findall(r'((\w)\2{2,})', s):
                s = s.replace(same_char[0], same_char[1])
            for punct in re.findall(r'(><.-「」【】『』[-/\\\\()!"+,&?\']{2,})',s):
                s = s.replace(punct, punct[0])
            s = re.sub("[-「」【】『』] ","",s)
            sentences.append(s)
        return sentences

inputData = sys.argv[1]
inputDictionary = sys.argv[2]
output = sys.argv[3]
jieba.set_dictionary(inputDictionary)
from gensim.models import Word2Vec
emb_model = Word2Vec.load('emb.model')

sentencesT = load_dataX(inputData)
sentencesT = [list(jieba.cut(s, cut_all=False)) for s in sentencesT]
from keras.preprocessing.sequence import pad_sequences
test_sequences = []
for i, s in enumerate(sentencesT):
    toks = []
    for w in s:
        try:
            toks.append(emb_model.wv.vocab[w].index + 1)
        except:
            toks.append(0)
    test_sequences.append(toks)
max_length = 100
test_sequences = pad_sequences(test_sequences, maxlen=max_length)

from keras.models import load_model
model = load_model('./model.h5')
testY = model.predict(test_sequences)
testY[testY >= 0.5] = 1
testY[testY < 0.5] = 0
testY = np.reshape(testY, 80000)
testY = testY.astype(int)
index = [i for i in range(0,len(testY))]
outCsv = pd.DataFrame({'id' : index, 'label' : testY})
outCsv.to_csv(output, index=0)
