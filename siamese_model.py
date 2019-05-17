from keras import backend as K
from keras import *
import numpy as np 
import os
from data_generator import *
from keras.layers import *
from keras import initializers
from keras.initializers import *
from keras.regularizers import l2




def get_siamese_model(n):
	"""
	model architecture

	Parameters:
	---------------------------------
	n: number of pairs to be sampled from each folder

	Returns:
	---------------------------------
	siamese_net : instance of the model

	"""
	# left, right, y = generator(n)

	# left = left.reshape(left.shape[0],left.shape[1],1)
	# right = right.reshape(right.shape[0],right.shape[1],1)
	input_shape = (16000,1)

	left_input = Input(input_shape)
	right_input = Input(input_shape)

	

	model = Sequential()
	model.add(Conv1D(12, 3, activation='relu', input_shape=input_shape
	           , kernel_regularizer=l2(.01)))
	model.add(MaxPooling1D())
	model.add(Conv1D(12,3, activation='relu',
	              kernel_regularizer=l2(.01)))
	model.add(MaxPooling1D())
	model.add(Conv1D(12,3, activation='relu',  kernel_regularizer=l2(.01)))
	model.add(MaxPooling1D())
	model.add(Conv1D(12,4, activation='relu',  kernel_regularizer=l2(.01)))
	model.add(Flatten())
	model.add(Dense(50, activation='sigmoid',
	           kernel_regularizer=l2(1e-3)))

	encoded_l = model(left_input)
	encoded_r = model(right_input)


	L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
	L1_distance = L1_layer([encoded_l, encoded_r])

	prediction = Dense(1,activation='sigmoid')(L1_distance)

	siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

	return siamese_net


def run_model(n):
	model = get_siamese_model(n)
	model.summary()

	# optimizer = Adam(lr = 0.00006)

	model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

	model.fit_generator(generator(n),steps_per_epoch=30, epochs=300,validation_data=get_test_set(10),validation_steps = 10)



if __name__ == '__main__':
	run_model(10)

    

