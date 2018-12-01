import numpy as np
import pandas as pd
from keras.models import Sequential  
from keras.layers.normalization import BatchNormalization
from keras.callbacks import History ,ModelCheckpoint , EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import sys

def normalize(x):
	x = x/255.
	return x

def cutTrain(x):
	train = x[0:26000]
	valid = x[26000::]
	return train, valid

def model1():
	model = Sequential()
	model.add(Conv2D(64,input_shape=input_shape, kernel_size=(5, 5), padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.3))

	model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.35))

	model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.4))
	model.add(Flatten())

	model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.7))
	model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.7))
	model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	check_save  = ModelCheckpoint("cnnModel/model1/model-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True)
	model.fit_generator(
				datagen.flow(trainX, trainY, batch_size=batch_size), 
				steps_per_epoch=len(trainX)//batch_size,
				validation_data=(validX, validY),
				epochs=epochs, callbacks=[check_save])
	model.save("cnnModel/model1/model1.h5")

def model2():
	model = Sequential()
	model.add(Conv2D(64,input_shape=input_shape, kernel_size=(5, 5), padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.3))

	model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.35))

	model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.4))
	model.add(Flatten())

	model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	check_save  = ModelCheckpoint("cnnModel/model1/model-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True)
	model.fit_generator(
				datagen.flow(trainX, trainY, batch_size=batch_size), 
				steps_per_epoch=len(trainX)//batch_size,
				validation_data=(validX, validY),
				epochs=epochs, callbacks=[check_save])
	model.save("cnnModel/model2/model2.h5")

def model3():
	model = Sequential()
	model.add(Conv2D(64,input_shape=input_shape,kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(BatchNormalization())
	model.add(Conv2D(64, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.3))

	model.add(Conv2D(256, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.35))

	model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.4))
	model.add(Flatten())

	model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.7))
	model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.7))
	model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	hist = History()
	check_save  = ModelCheckpoint("cnnModel/model3/model-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True)
	model.fit_generator(
				datagen.flow(trainX, trainY, batch_size=batch_size), 
				steps_per_epoch=len(trainX)//batch_size,
				validation_data=(validX, validY),
				epochs=epochs, callbacks=[check_save,hist])
	model.save("cnnModel/model3/model.h5")

def model4():
	model = Sequential()
	model.add(Conv2D(32,input_shape=input_shape, activation='relu',kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(Conv2D(32,input_shape=input_shape, activation='relu',kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.3))

	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.35))

	model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.4))

	model.add(Flatten())
	model.add(Dense(1024, activation='relu', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.7))
	model.add(Dense(1024, activation='relu', kernel_initializer='glorot_normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.7))
	model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	hist = History()
	check_save  = ModelCheckpoint("cnnModel/model4/model-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True)
	model.fit_generator(
				datagen.flow(trainX, trainY, batch_size=batch_size), 
				steps_per_epoch=len(trainX)//batch_size,
				validation_data=(validX, validY),
				epochs=epochs, callbacks=[check_save,hist])
	model.save("cnnModel/model4/model.h5")


if __name__ == '__main__':
	train = sys.argv[1]
	data = pd.read_csv(train)
	feature = np.array([row.split(" ") for row in data["feature"].tolist()],dtype=np.float32)
	feature = np.reshape(feature, (28709, 48,48,1))
	raw = np.array(data['label'])
	label = []
	for num in raw:
		temp = [0,0,0,0,0,0,0]
		temp[num] = 1
		label.append(temp)
	label = np.array(label)

	feature = normalize(feature)
	trainX, validX = cutTrain(feature)
	trainY, validY = cutTrain(label)
	datagen = ImageDataGenerator(
		rotation_range=30.0,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=[0.8, 1.2],
		horizontal_flip=True,
		vertical_flip=False)

	batch_size = 260
	epochs = 400
	input_shape = (48,48,1)
	model1()
	model2()
	model3()
	model4()
