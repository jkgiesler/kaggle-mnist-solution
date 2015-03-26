from pandas import DataFrame, Series
import cPickle as pickle
import lasagne
from lasagne import layers
from lasagne.layers import cuda_convnet
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.utils import shuffle

import theano
import theano.tensor as T
import numpy 
import pickle

#Constants
NUM_EPOCHS = 1
BATCH_SIZE = 600
LEARNING_RATE = 0.01
MOMENTUM = 0.9
dimshuffle = True

##rewritten MNIST conv. neural net using daniel nouri's face recognition tutorial

def _load_data():
	train_df = DataFrame.from_csv('~/Downloads/digits_train.csv', index_col=False)
	test_df = DataFrame.from_csv('~/Downloads/digits_test.csv', index_col=False)
	##divide by 255 as data is pixel stuff.
	train_set = [train_df.values[0:,1:]/ 255.,train_df.values[0:,0]]
	test_set = test_df.values / 255.
	return_me = [train_set,test_set]
	return return_me

def load_data():
	#no need for validation.
	data = _load_data()
	X_train,Y_train = data[0]
	X_test = data[1]

	#this has something to do with preparation for convolution
	#read up on this probably

	X_train = X_train.reshape((X_train.shape[0],1,28,28))
	X_test = X_test.reshape((X_test.shape[0],1,28,28))
	#X_train = X_train.astype(numpy.float32)
	#X_test = X_train.astype(numpy.float32)
	#Y_train = Y_train.astype(numpy.int32)
	return dict(
		X_train=lasagne.utils.floatX(X_train),
		y_train=Y_train.astype(numpy.int32),
		X_test=lasagne.utils.floatX(X_test),
		num_examples_train=X_train.shape[0],
		num_examples_test=X_test.shape[0],
		input_height=X_train.shape[2],
		input_width=X_train.shape[3],
		output_dim=10,
    )
dataset = load_data()
#instanciate a NeuralNet class
convNet = NeuralNet(
	##layer constructor
	layers=[
	('input',layers.InputLayer),
	('conv1',cuda_convnet.Conv2DCCLayer),
	('pool1',cuda_convnet.MaxPool2DCCLayer),
	('conv2',cuda_convnet.Conv2DCCLayer),
	('pool2',cuda_convnet.MaxPool2DCCLayer),
	('hidden1',layers.DenseLayer),
	('dropout1',layers.DropoutLayer),
	('output',layers.DenseLayer),
	],

	##layer parameters None specifies input layer may be of different lengths
	input_shape = (None,1,dataset['input_width'],dataset['input_height']),
	conv1_num_filters =32,
	conv1_filter_size=(5,5),
	conv1_nonlinearity=lasagne.nonlinearities.rectify,
	conv1_W=lasagne.init.Uniform(),
	conv1_dimshuffle=dimshuffle,

	pool1_ds=(2,2),
	pool1_dimshuffle=dimshuffle,

	conv2_num_filters=32,
	conv2_filter_size=(5,5),
	conv2_nonlinearity=lasagne.nonlinearities.rectify,
	conv2_W=lasagne.init.Uniform(),
	conv2_dimshuffle=dimshuffle,

	pool2_ds=(2,2),
	pool2_dimshuffle=dimshuffle,

	hidden1_num_units=256,
	hidden1_nonlinearity=lasagne.nonlinearities.rectify,
	hidden1_W=lasagne.init.Uniform(),
	
	dropout1_p=.5,

	output_num_units = 10, ## should be the same as dataset['output_dim'] 
	output_nonlinearity=lasagne.nonlinearities.softmax,
	output_W=lasagne.init.Uniform(),

	#NN parameters
	update_learning_rate = LEARNING_RATE,
	update_momentum = MOMENTUM,
	regression = False,
	#batch_iterator_train=FlipBatchIterator(batch_size=BATCH_SIZE),
	max_epochs = NUM_EPOCHS,
	verbose=1,
)


convNet.fit(dataset['X_train'],dataset['y_train'])


with open('covNet.pickle','wb') as f:
	pickle.dump(convNet,f,-1)

y_pred = convNet.predict(dataset['X_test'])
y_pred = Series(y_pred)
image_ids = Series(numpy.arange(1,len(y_pred)+1))

submission = DataFrame([image_ids,y_pred]).T
submission.columns = ['ImageId','Label']
submission.to_csv('submission_test_cuda.csv',index=False)
