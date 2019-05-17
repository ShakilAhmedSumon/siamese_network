import librosa
import numpy as np 
import os
import random


def get_similar_pairs(dir,n):
	"""
	Returns n similar pairs

	Parameters:
	----------------------------------
	dir : directory from where the audios should be sampled
	n   : number of samples

	Returns:
	----------------------------------
	left  : list of audio files
	right : list of audio files

	"""

	left = []
	right = []
	y = []

	for i in range(n):
		number_1 = random.randint(20,199)
		number_2 = random.randint(20,199)

		audio_list = os.listdir(dir + '//')

		audio_1,sr = librosa.load(dir + '//'+ audio_list[number_1], mono = True, sr = 16000)
		audio_2,sr = librosa.load(dir + '//'+ audio_list[number_2], mono = True, sr = 16000)

		pad_width_1 = 16000 - len(audio_1)
		pad_width_2 = 16000 - len(audio_2)

		if(len(audio_1)<16000):
			audio_1 = np.pad(audio_1,(0,pad_width_1),mode = 'constant')

		if(len(audio_2)<16000):
			audio_2 = np.pad(audio_2,(0,pad_width_2),mode = 'constant')

		left.append(audio_1)
		right.append(audio_2)
		y.append(1)

	return left, right, y

def get_similar_test_pairs(dir,n):
	"""
	Returns n similar pairs

	Parameters:
	----------------------------------
	dir : directory from where the audios should be sampled
	n   : number of samples

	Returns:
	----------------------------------
	left  : list of audio files
	right : list of audio files

	"""
	path = 'test/'

	left = []
	right = []
	y = []

	for i in range(n):
		number_1 = random.randint(1,19)
		number_2 = random.randint(1,19)

		audio_list = os.listdir(dir + '//')

		audio_1,sr = librosa.load(dir + '//'+ audio_list[number_1], mono = True, sr = 16000)
		audio_2,sr = librosa.load(dir + '//'+ audio_list[number_2], mono = True, sr = 16000)

		pad_width_1 = 16000 - len(audio_1)
		pad_width_2 = 16000 - len(audio_2)

		if(len(audio_1)<16000):
			audio_1 = np.pad(audio_1,(0,pad_width_1),mode = 'constant')

		if(len(audio_2)<16000):
			audio_2 = np.pad(audio_2,(0,pad_width_2),mode = 'constant')

		left.append(audio_1)
		right.append(audio_2)
		y.append(1)

	return left, right, y


def get_dissimilar_pairs(dir1, dir2, n):
	"""
	Returns n dissimilar pairs

	Parameters:
	----------------------------------
	dir1 : directory from where the audios should be sampled
	dir2 : directory from where the audios should be sampled
	n    : number of samples

	Returns:
	----------------------------------
	left  : list of audio files from directory1
	right : list of audio files from directory2

	"""

	left = []
	right = []
	y = []

	for i in range(n):
		number_1 = random.randint(20,199)
		number_2 = random.randint(20,199)

		audio_list_1 = os.listdir(dir1 + '//')
		audio_list_2 = os.listdir(dir2 + '//')

		audio_1,sr = librosa.load(dir1 + '//' + audio_list_1[number_1], mono = True, sr = 16000)
		audio_2,sr = librosa.load(dir2 + '//' + audio_list_2[number_2], mono = True, sr = 16000)

		pad_width_1 = 16000 - len(audio_1)
		pad_width_2 = 16000 - len(audio_2)

		if(len(audio_1)<16000):
			audio_1 = np.pad(audio_1,(0,pad_width_1),mode = 'constant')

		if(len(audio_2)<16000):
			audio_2 = np.pad(audio_2,(0,pad_width_2),mode = 'constant')


		left.append(audio_1)
		right.append(audio_2)
		y.append(0)

	return left, right, y

def get_dissimilar_test_pairs(dir1, dir2, n):
	"""
	Returns n dissimilar pairs

	Parameters:
	----------------------------------
	dir1 : directory from where the audios should be sampled
	dir2 : directory from where the audios should be sampled
	n    : number of samples

	Returns:
	----------------------------------
	left  : list of audio files from directory1
	right : list of audio files from directory2

	"""
	path = 'test/'

	left = []
	right = []
	y = []

	for i in range(n):
		number_1 = random.randint(1,19)
		number_2 = random.randint(1,19)

		audio_list_1 = os.listdir(dir1 + '//')
		audio_list_2 = os.listdir(dir2 + '//')

		audio_1,sr = librosa.load(dir1 + '//' + audio_list_1[number_1], mono = True, sr = 16000)
		audio_2,sr = librosa.load(dir2 + '//' + audio_list_2[number_2], mono = True, sr = 16000)

		pad_width_1 = 16000 - len(audio_1)
		pad_width_2 = 16000 - len(audio_2)

		if(len(audio_1)<16000):
			audio_1 = np.pad(audio_1,(0,pad_width_1),mode = 'constant')

		if(len(audio_2)<16000):
			audio_2 = np.pad(audio_2,(0,pad_width_2),mode = 'constant')


		left.append(audio_1)
		right.append(audio_2)
		y.append(0)

	return left, right, y

def generator(n):
	"""
	generates the dataset

	Parameters:
	---------------------------------
	n : number of pairs to be sampled from each folder

	Returns:
	--------------------------------
	left  : array containing similar and dissimilar audio files
	right : array containing similar and dissimilar audio files
	y     : class labels of the similar and dissimilar pairs

	"""
	
	while True:
		folders = ['cat', 'dog', 'down', 'house','left']

		left  = np.zeros((5,16000))
		right = np.zeros((5,16000))
		y = np.zeros(5)


		for folder in folders:
			left_similar,right_similar,y_similar = get_similar_pairs(folder,n)

			left_similar = np.asarray(left_similar)
			# print(left_similar.shape)
			right_similar = np.asarray(right_similar)
			y_similar = np.asarray(y_similar)

			left  = np.concatenate((left,left_similar),axis = 0)
			right = np.concatenate((right,right_similar),axis = 0)
			y = np.concatenate((y,y_similar))

			for path in folders:
				if path != folder:
					left_dissimilar,right_dissimilar,y_dissimilar = get_dissimilar_pairs(folder,path,int(n/2))

					left_dissimilar = np.asarray(left_dissimilar)
					# print(left_dissimilar.shape)
					right_dissimilar = np.asarray(right_dissimilar)
					# print(right_dissimilar.shape)
					y_dissimilar = np.asarray(y_dissimilar)

					left  = np.concatenate((left,left_dissimilar),axis = 0)
					right  = np.concatenate((right,right_dissimilar),axis = 0)
					y = np.concatenate((y,y_dissimilar))



		left = left[5:]
		right = right[5:]
		y = y[5:]
		left_mean = np.mean(left)
		left_std = np.std(left)
		left = left - left_mean
		left = left/left_std

		right_mean = np.mean(right)
		right_std = np.std(right)
		right = right - right_mean
		right = right/right_std

		left = left.reshape(left.shape[0],left.shape[1],1)
		right = right.reshape(right.shape[0],right.shape[1],1)

		# print(left.shape)
		# return left,right,y
		# yield (left,right,y)
		yield ({'input_1': left, 'input_2': right}, {'dense_2': y})


def get_test_set(n):
	while True:
		folders = [ 'cat','dog', 'down', 'house','left']

		left  = np.zeros((5,16000))
		right = np.zeros((5,16000))
		y = np.zeros(5)


		for folder in folders:
			left_similar,right_similar,y_similar = get_similar_test_pairs(folder,n)

			left_similar = np.asarray(left_similar)
			# print(left_similar.shape)
			right_similar = np.asarray(right_similar)
			y_similar = np.asarray(y_similar)

			left  = np.concatenate((left,left_similar),axis = 0)
			right = np.concatenate((right,right_similar),axis = 0)
			y = np.concatenate((y,y_similar))

			for path in folders:
				if path != folder:
					left_dissimilar,right_dissimilar,y_dissimilar = get_dissimilar_test_pairs(folder,path,int(n/2))

					left_dissimilar = np.asarray(left_dissimilar)
					# print(left_dissimilar.shape)
					right_dissimilar = np.asarray(right_dissimilar)
					# print(right_dissimilar.shape)
					y_dissimilar = np.asarray(y_dissimilar)

					left  = np.concatenate((left,left_dissimilar),axis = 0)
					right  = np.concatenate((right,right_dissimilar),axis = 0)
					y = np.concatenate((y,y_dissimilar))



		left = left[5:]
		right = right[5:]
		y = y[5:]

		left_mean = np.mean(left)
		left_std = np.std(left)
		left = left - left_mean
		left = left/left_std

		right_mean = np.mean(right)
		right_std = np.std(right)
		right = right - right_mean
		right = right/right_std
		
		left = left.reshape(left.shape[0],left.shape[1],1)
		right = right.reshape(right.shape[0],right.shape[1],1)

		# print(left.shape)
		# return left,right,y
		# yield (left,right,y)
		yield ({'input_1': left, 'input_2': right}, {'dense_2': y})


if __name__ == '__main__':
	left , right, y = generator(50)
	print(len(left), len(right), len(y))
	