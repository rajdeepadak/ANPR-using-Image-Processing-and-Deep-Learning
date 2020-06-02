from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation,Dropout,Flatten,BatchNormalization
from keras.models import Model
import cv2, os
import numpy as np


labels = {0: '0',1: '1', 2: '2', 3: '3', 4: '4', 5: '5',6: '6',7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}

 # labels = {0: '-',1: '0',2: '1', 3: '2', 4: '3', 5: '4', 6: '5',7: '6',8: '7', 9: '8', 10: '9', 11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E',
 # 16: 'F', 17: 'G', 18: 'H', 19: 'I', 20: 'J', 21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q', 28: 'R',
 # 29: 'S', 30: 'T', 31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y', 36: 'Z'}


def get_model():

	inputs = Input(shape=(28,28,1))

	conv1 = Conv2D(16, kernel_size=(3, 3),activation='relu')(inputs)
	conv2 = Conv2D(32, kernel_size=(3, 3),activation='relu')(conv1)
	pool = MaxPooling2D(pool_size=(2,2))(conv2)
	dropout1 = Dropout(0.25)(pool)
	flatten = Flatten()(dropout1)
	dense1 = Dense(64, activation='relu')(flatten)
	dropout2 = Dropout(0.5)(dense1)
	output = Dense(36, activation='softmax')(dropout2)

	model = Model(inputs=inputs, outputs=output)

	return model

def predict_char(img):
	img = img.reshape(1,28,28,1)
	img = img/255.0

	pred = model.predict(img)

	idx = np.argmax(pred)

	char_class = labels[idx]

	return char_class


model = get_model()
print(model.summary())
model.load_weights('char_weights--1.0000.hdf5')

# img = cv2.imread('class_H_17.jpg',0)

# char = predict_char(img)
# print('output:',char)
