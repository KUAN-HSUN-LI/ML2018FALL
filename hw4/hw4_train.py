import pandas as pd
import numpy as np
import re
import sys

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
            s = re.sub("[-「」【】『』]","",s)
            sentences.append(s)
        return sentences

def load_dataY(path):
    labels = pd.read_csv(path)['label']
    return list(labels)

inputTrainX = sys.argv[1]
inputTrainY = sys.argv[2]
inputTestX = sys.argv[3]
dictionary = sys.argv[4]

sentencesX = load_dataX(inputTrainX)
sentencesT = load_dataX(inputTestX)
sentences = sentencesX + sentencesT
labels = load_dataY(inputTrainY)

import jieba

jieba.set_dictionary(dictionary)
sentences = [list(jieba.cut(s, cut_all=False)) for s in sentences]
sentencesX = [list(jieba.cut(s, cut_all=False)) for s in sentencesX]

from gensim.models import Word2Vec
emb_dim = 100
# Train Word2Vec model
emb_model = Word2Vec(sentences, size=emb_dim, window = 5, min_count = 3, workers = 8, iter = 20, negative = 10)
num_words = len(emb_model.wv.vocab) + 1  # +1 for OOV words
print(num_words)
# Create embedding matrix (For Keras)
emb_matrix = np.zeros((num_words, emb_dim), dtype=float)
for i in range(num_words - 1):
    v = emb_model.wv[emb_model.wv.index2word[i]]
    emb_matrix[i+1] = v   # Plus 1 to reserve index 0 for OOV words
emb_model.save('emb123.model')

max_length = 100
i = 0
while i != len(sentencesX):
    if len(sentencesX[i]) > max_length:
        sentencesX.pop(i)
        labels.pop(i)
        i -= 1
    i += 1

from keras.preprocessing.sequence import pad_sequences

# Convert words to index
train_sequences = []
for i, s in enumerate(sentencesX):
    toks = []
    for w in s:
        try:
            toks.append(emb_model.wv.vocab[w].index + 1)
        except:
            toks.append(0)
    train_sequences.append(toks)


# Pad sequence to same length
train_sequences = pad_sequences(train_sequences, maxlen=max_length)
split_sit = int(len(train_sequences) * 0.9)
(X_train, Y_train)  = (train_sequences[:split_sit], np.array(labels[:split_sit]))
(X_val, Y_val) = (train_sequences[split_sit:], np.array(labels[split_sit:]))

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Bidirectional, GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import History
from keras.layers.normalization import BatchNormalization

def rnn1():
	model = Sequential()
	model.add(Embedding(num_words,
						emb_dim,
						weights=[emb_matrix],
						input_length=max_length,
						trainable=True))
	model.add(Bidirectional(LSTM(256,activation="tanh",  dropout=0.3, recurrent_dropout=0.3, return_sequences = True, kernel_initializer='Orthogonal')))
	model.add(Bidirectional(LSTM(128,activation="tanh",  dropout=0.3, recurrent_dropout=0.3, return_sequences = False, kernel_initializer='Orthogonal')))
	model.add(BatchNormalization())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.summary()

	adam = Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
	model.compile(loss='binary_crossentropy',
				optimizer=adam,
				metrics=['accuracy'])

	csv_logger = CSVLogger('training.log')
	checkpoint = ModelCheckpoint(filepath="./finish-1/modeltest-{epoch:05d}-{val_acc:.5f}.h5",
								verbose=1,
								save_best_only=True,
								monitor='val_acc',
								mode='max')

	hist = History()
	epochs = 10
	batch_size = 256
	model.fit(X_train, Y_train, 
			validation_data=(X_val, Y_val),
			epochs=epochs, 
			batch_size=batch_size,
			callbacks=[hist, checkpoint, csv_logger])

def rnn2():
	model = Sequential()
	model.add(Embedding(num_words,
						emb_dim,
						weights=[emb_matrix],
						input_length=max_length,
						trainable=True))
						
	model.add(Bidirectional(GRU(256,activation="tanh",  dropout=0.3, recurrent_dropout=0.3, return_sequences = True, kernel_initializer='Orthogonal')))
	model.add(Bidirectional(GRU(128,activation="tanh",  dropout=0.3, recurrent_dropout=0.3, return_sequences = False, kernel_initializer='Orthogonal')))
	model.add(BatchNormalization())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.summary()

	adam = Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
	model.compile(loss='binary_crossentropy',
				optimizer=adam,
				metrics=['accuracy'])

	csv_logger = CSVLogger('training.log')
	checkpoint = ModelCheckpoint(filepath="./finish-2/modeltest-{epoch:05d}-{val_acc:.5f}.h5",
								verbose=1,
								save_best_only=True,
								monitor='val_acc',
								mode='max')
	hist = History()
	epochs = 10
	batch_size = 256
	model.fit(X_train, Y_train, 
			validation_data=(X_val, Y_val),
			epochs=epochs, 
			batch_size=batch_size,
			callbacks=[hist, checkpoint, csv_logger])

def rnn3():
	model = Sequential()
	model.add(Embedding(num_words,
						emb_dim,
						weights=[emb_matrix],
						input_length=max_length,
						trainable=False))
						
	model.add(Bidirectional(GRU(256,activation="tanh",  dropout=0.3, recurrent_dropout=0.3, return_sequences = True, kernel_initializer='Orthogonal')))
	model.add(Bidirectional(GRU(128,activation="tanh",  dropout=0.3, recurrent_dropout=0.3, return_sequences = False, kernel_initializer='Orthogonal')))
	model.add(BatchNormalization())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.summary()

	adam = Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
	model.compile(loss='binary_crossentropy',
				optimizer=adam,
				metrics=['accuracy'])

	csv_logger = CSVLogger('training.log')
	checkpoint = ModelCheckpoint(filepath="./finish-3/modeltest-{epoch:05d}-{val_acc:.5f}.h5",
								verbose=1,
								save_best_only=True,
								monitor='val_acc',
								mode='max')
	hist = History()
	epochs = 10
	batch_size = 256
	model.fit(X_train, Y_train, 
			validation_data=(X_val, Y_val),
			epochs=epochs, 
			batch_size=batch_size,
			callbacks=[hist, checkpoint, csv_logger])

def rnn4():
	model = Sequential()
	model.add(Embedding(num_words,
						emb_dim,
						weights=[emb_matrix],
						input_length=max_length,
						trainable=False))
						
	model.add(Bidirectional(LSTM(256,activation="tanh",  dropout=0.3, recurrent_dropout=0.3, return_sequences = True, kernel_initializer='Orthogonal')))
	model.add(Bidirectional(LSTM(128,activation="tanh",  dropout=0.3, recurrent_dropout=0.3, return_sequences = False, kernel_initializer='Orthogonal')))
	model.add(BatchNormalization())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.summary()

	adam = Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
	model.compile(loss='binary_crossentropy',
				optimizer=adam,
				metrics=['accuracy'])


	csv_logger = CSVLogger('training.log')
	checkpoint = ModelCheckpoint(filepath="./finish-4/modeltest-{epoch:05d}-{val_acc:.5f}.h5",
								verbose=1,
								save_best_only=True,
								monitor='val_acc',
								mode='max')
	hist = History()
	epochs = 10
	batch_size = 256
	model.fit(X_train, Y_train, 
			validation_data=(X_val, Y_val),
			epochs=epochs, 
			batch_size=batch_size,
			callbacks=[hist, checkpoint, csv_logger])

def rnn5():
	model = Sequential()
	model.add(Embedding(num_words,
						emb_dim,
						weights=[emb_matrix],
						input_length=max_length,
						trainable=False))
						
	model.add(LSTM(256,activation="tanh",  dropout=0.3, recurrent_dropout=0.3, return_sequences = True, kernel_initializer='Orthogonal'))
	model.add(LSTM(128,activation="tanh",  dropout=0.3, recurrent_dropout=0.3, return_sequences = False, kernel_initializer='Orthogonal'))
	model.add(BatchNormalization())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.summary()

	adam = Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
	model.compile(loss='binary_crossentropy',
				optimizer=adam,
				metrics=['accuracy'])


	csv_logger = CSVLogger('training.log')
	checkpoint = ModelCheckpoint(filepath="./finish-5/modeltest-{epoch:05d}-{val_acc:.5f}.h5",
								verbose=1,
								save_best_only=True,
								monitor='val_acc',
								mode='max')

	hist = History()
	epochs = 10
	batch_size = 256
	model.fit(X_train, Y_train, 
			validation_data=(X_val, Y_val),
			epochs=epochs, 
			batch_size=batch_size,
			callbacks=[hist, checkpoint, csv_logger])

from keras.models import load_model
from keras.models import Model
from keras import layers
from keras.layers import Input, Dense

'''
def ensembleModels(ymodel):
    model_input = Input(shape=ymodel[0].input_shape[1:])
    yModels=[model(model_input) for model in ymodel] 
    yAvg=layers.average(yModels)
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')
    print (modelEns.summary())
    modelEns.save('ensemble.h5')


m1 = load_model('./finish-1/modeltest-00006-0.75506.h5')
m2 = load_model('./finish-2/modeltest-00004-0.75293.h5')
m3 = load_model('./finish-3/modeltest-00008-0.75213.h5')
m4 = load_model('./finish-4/modeltest-00010-0.75248.h5')
m5 = load_model('./finish-5/modeltest-00010-0.75142.h5')
ensembleModels(ymodel=[m1, m2, m3, m4, m5])
'''

if __name__ == "__main__":
	rnn1()
	rnn2()
	rnn3()
	rnn4()
	rnn5()
