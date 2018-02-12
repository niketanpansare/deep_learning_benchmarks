#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

# To check usage, use -h option.


import time
start = time.time()
data_loading = 0.0
import os, argparse, sys

# The framework elephas (as of Feb 5th, 2018) fails with TypeError: 'LabeledPoint' object is not iterable

parser=argparse.ArgumentParser("Deep Learning (DL) Benchmarks.")
parser.add_argument('--model', help='DL model. Supported values are: lenet, sentence_cnn_static. Default: lenet', type=str, default='lenet')
parser.add_argument('--data', help='Dataset to use for training/testing. Supported values are: mnist, imdb. Default: mnist', type=str, default='mnist')
parser.add_argument('--data_format', help='Data format. Supported values are: spark_df, numpy, scipy. Default: numpy', type=str, default='numpy')
parser.add_argument('--epochs', help='Number of epochs. Default: 10', type=int, default=10)
parser.add_argument('--batch_size', help='Batch size. Default: 64', type=int, default=64)
parser.add_argument('--num_gpus', help='Number of GPUs. Default: 0', type=int, default=0)
parser.add_argument('--framework', help='DL Framework. Supported values are: systemml, keras, tensorflow, bigdl. Default: systemml', type=str, default='systemml')
parser.add_argument('--precision', help='Floating point precision. Supported values are: single, double. Default: single', type=str, default='single')
parser.add_argument('--blas', help='Native BLAS. Supported values are: openblas, mkl, none. Default: openblas', type=str, default='openblas')
parser.add_argument('--phase', help='Training/testing phase. Supported values are: train, test. Default: train', type=str, default='train')
parser.add_argument('--codegen', help='Whether to apply codegen optimization. Supported values are: enabled, disabled. Default: disabled', type=str, default='disabled')
args=parser.parse_args()

config = {'display':100}

if args.precision != 'single' and args.precision != 'double':
        raise ValueError('Incorrect precision:' + args.precision)

if args.framework == 'systemml' or args.num_gpus == 0:
	# When framework is systemml, force any tensorflow allocation to happen on CPU to avoid GPU OOM
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = ''
elif args.num_gpus == 1:
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Hyperparameter for sentence CNN with imdb dataset
sentence_length, embedding_dim = 2494, 300 # Maximum sentence length in imdb is 2494
input_shapes = { 'mnist':(1,28,28), 'imdb':(1, sentence_length, embedding_dim) }
num_labels = {'mnist':10, 'imdb':2 }

t0 = time.time()
try:
	from pyspark import SparkContext
	sc = SparkContext()
	sc.parallelize([1, 2, 3, 4, 5]).count()
	if args.framework != 'elephas':
		# As elephas only support Spark 1.6 or less as of Feb 5th 2018
		from pyspark.sql import SparkSession
		spark = SparkSession.builder.getOrCreate()
		spark.createDataFrame([(1, 4), (2, 5), (3, 6)], ["A", "B"]).count()
except ImportError:
	raise ImportError('Make sure you are running with Spark. Hint: $SPARK_HOME/bin/spark-submit compare_framework.py -h')
spark_init_time = time.time() - t0

import math
import numpy as np
import keras
right_keras_version = '1.2.2' if args.framework == 'bigdl' else '2.1.3'
if keras.__version__ != right_keras_version:
	print('WARNING: Incorrect keras version:' + str(keras.__version__) + '. Expected keras version:' + right_keras_version)
from keras.models import Sequential, Model
from keras.layers import Input, Dense, MaxPooling2D, Dropout, Flatten, Reshape
from keras import backend as K
from keras.models import Model
from keras.utils import np_utils
from keras.backend.tensorflow_backend import set_session
from keras import regularizers
from keras.preprocessing import sequence
if args.framework == 'bigdl':
	# bigdl only supports keras 1.2
	from keras.layers import Convolution2D, merge
else:
	from keras.layers import Conv2D, Concatenate

try:
	import tensorflow as tf
except ImportError:
        raise ImportError('Make sure you have installed tensorflow')


# For fair comparison
if args.framework == 'tensorflow':
	tf_config = tf.ConfigProto()
	#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
	tf_config.gpu_options.allow_growth = True
	if args.codegen == 'enabled':
		tf_config.graph_options.optimizer_options.global_jit_level = K.tf.OptimizerOptions.ON_1
	else:
		tf_config.graph_options.optimizer_options.global_jit_level = 0
	set_session(tf.Session(config=tf_config))

if args.precision == 'double' and args.framework == 'tensorflow':
	K.set_floatx('float64')
	#print('double precision is not supported in Keras. See https://stackoverflow.com/questions/48552508/running-keras-with-double-precision-fails/')

def get_keras_input_shape(input_shape):
	#if K.image_data_format() == 'channels_first':
	#	return input_shape
	#else:
	#	return ( input_shape[1], input_shape[2], input_shape[0] )
	return ( input_shape[1], input_shape[2], input_shape[0] )

def conv2d(nb_filter, nb_row, nb_col, activation='relu', padding='same', input_shape=None):
	if input_shape != None:
		if args.framework == 'bigdl':
			return Convolution2D(nb_filter, nb_row, nb_col, activation=activation, input_shape=input_shape, W_regularizer=regularizers.l2(0.01), border_mode=padding)
		else:
			return Conv2D(nb_filter, kernel_size=(nb_row, nb_col), activation=activation, input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01), padding=padding)
	else:
		if args.framework == 'bigdl':
                        return Convolution2D(nb_filter, nb_row, nb_col, activation=activation, W_regularizer=regularizers.l2(0.01), border_mode=padding)
                else:
                        return Conv2D(nb_filter, kernel_size=(nb_row, nb_col), activation=activation, kernel_regularizer=regularizers.l2(0.01), padding=padding)

def dense(num_out, activation='relu'):
	if args.framework == 'bigdl':
		return Dense(num_out, activation=activation, W_regularizer=regularizers.l2(0.01))
	else:
		return Dense(num_out, activation=activation, kernel_regularizer=regularizers.l2(0.01))

def concat(conv_ngrams):
	if len(conv_ngrams) == 1:
		return conv_ngrams[0]
	if args.framework == 'bigdl':
		return merge(conv_ngrams, mode='concat', concat_axis=1)
	else:
		return Concatenate()(conv_ngrams)

def reshape_for_bigdl(inputL=None):
        input_shape = get_keras_input_shape(input_shapes[args.data]) # if args.framework == 'tensorflow' else input_shapes[args.data]
        # As per https://github.com/intel-analytics/BigDL/blob/master/pyspark/bigdl/models/lenet/lenet5.py
        if inputL == None:
                return Reshape(input_shape, input_shape=(1, input_shape[0]*input_shape[1]*input_shape[2]))
        else:
                return Reshape(input_shape)(inputL)

def get_keras_model():
	if args.model == 'lenet':
		# Sequential model
		keras_model = Sequential()
		if args.framework == 'bigdl':
			keras_model.add(reshape_for_bigdl())
		keras_model.add(conv2d(32, 5, 5, input_shape=get_keras_input_shape(input_shapes[args.data])))
		keras_model.add(MaxPooling2D(pool_size=(2, 2)))
		keras_model.add(conv2d(64, 5, 5))
		keras_model.add(MaxPooling2D(pool_size=(2, 2)))
		keras_model.add(Flatten())
		keras_model.add(dense(512))
		keras_model.add(Dropout(0.5))
		keras_model.add(dense(10, activation='softmax'))
	elif args.model == 'sentence_cnn_static':
		# Functional model
		conv_ngrams = []
		nb_filters = 100
		inputL = Input(shape=get_keras_input_shape(input_shapes[args.data]))
                inputL1 = reshape_for_bigdl(inputL) if args.framework == 'bigdl' else inputL
		kernel_h = [2, 3, 4, 5]
		pool_h = [ sentence_length - k for k in kernel_h ]
		for i in range(len(kernel_h)):
			conv = conv2d(nb_filters, kernel_h[i], embedding_dim, activation='relu', padding='valid')(inputL1)
			conv = MaxPooling2D(pool_size=(pool_h[i], 1))(conv)
			conv = Flatten()(conv)
			conv_ngrams.append(conv)
		z = concat(conv_ngrams)
		z = dense(500, activation='relu')(z)
		z = Dropout(0.5)(z)
		model_output = dense(2, activation='sigmoid')(z)
		keras_model = Model(inputL, model_output)
	else:
		raise ValueError('Unsupported model:' + str(args.model))
	#if type(keras_model) == keras.models.Sequential:
	#	# Convert the sequential model to functional model
	#	if keras_model.model is None:
	#		keras_model.build()
	#	keras_model = keras_model.model
	return keras_model

t0 = time.time()
y = None
if args.data_format == 'numpy' or args.data_format == 'scipy':
	from sklearn.datasets import load_svmlight_file
	in_shape = input_shapes[args.data]
	X, y = load_svmlight_file(args.data + '.libsvm', n_features=in_shape[0]*in_shape[1]*in_shape[2], zero_based=False)
	y = y if args.framework == 'bigdl' else y-1 # convert one-based labels to zero-based
	if args.data_format == 'numpy':
		X = X.toarray()
		if args.precision == 'double':
			X = X.astype(np.float64)
elif args.data_format == 'spark_df':
	from pyspark.mllib.util import MLUtils
	in_shape = input_shapes[args.data]
	X = MLUtils.convertVectorColumnsToML(MLUtils.loadLibSVMFile(sc, args.data + '.libsvm', multiclass=False if args.data == 'imdb' else True, numFeatures=in_shape[0]*in_shape[1]*in_shape[2]).toDF())
	#X = spark.read.format("libsvm").load(args.data + '.libsvm')
else:
	raise ValueError('Unsupported data format:' + rgs.data_format)
data_loading = data_loading + time.time() - t0

epochs = int(args.epochs)
batch_size = int(args.batch_size)
num_samples = X.shape[0] if hasattr(X, 'shape') else X.count()
max_iter = int(epochs*math.ceil(num_samples/batch_size))
display = int(config['display'])

def get_framework_model(framework):
	keras_model = get_keras_model()
	loss = 'categorical_crossentropy' if args.data =='mnist' else 'mean_squared_error'
	if not (framework == 'tensorflow' and args.num_gpus >= 2):
		keras_model.compile(loss=loss, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.95, decay=5e-4, nesterov=True))
	if framework == 'systemml':
		from systemml.mllearn import Keras2DML
		load_keras_weights = True if args.phase == 'test' else False
		sysml_model = Keras2DML(spark, keras_model, input_shape=input_shapes[args.data], batch_size=batch_size, max_iter=max_iter, test_iter=0, display=display, load_keras_weights=load_keras_weights)
		sysml_model.setStatistics(True).setStatisticsMaxHeavyHitters(100)
		sysml_model.setConfigProperty("sysml.gpu.sync.postProcess", "false")
		#sysml_model.set(debug=True)
		#sysml_model.setConfigProperty("sysml.stats.finegrained", "true")
		#sysml_model.setConfigProperty("sysml.gpu.eager.cudaFree", "true")
		# From configuration:
		sysml_model.setConfigProperty("sysml.native.blas", args.blas)
		sysml_model.setConfigProperty("sysml.floating.point.precision", args.precision)
		if args.codegen == 'enabled':
			sysml_model.setConfigProperty("sysml.codegen.enabled", "true").setConfigProperty("sysml.codegen.optimizer", "fuse_no_redundancy")
		# .setConfigProperty("sysml.codegen.plancache", "true")
		if args.num_gpus >= 1:
			sysml_model.setGPU(True).setForceGPU(True)
		if args.num_gpus > 1:
			sysml_model.set(train_algo="allreduce_parallel_batches", parallel_batches=args.num_gpus)
		return sysml_model
	elif framework == 'tensorflow':
		if args.num_gpus >= 2:
			from keras.utils import multi_gpu_model
			keras_model = multi_gpu_model(keras_model, gpus=args.num_gpus)
			keras_model.compile(loss=loss, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.95, decay=5e-4, nesterov=True))
		return keras_model
	elif  framework == 'elephas':
		from elephas.spark_model import SparkModel
		from elephas import optimizers as elephas_optimizers
		optim = elephas_optimizers.SGD(lr=0.01, momentum=0.95, decay=5e-4, nesterov=True)
		spark_model = SparkModel(sc, keras_model, optimizer=optim,mode='synchronous') #, num_workers=2)
		return spark_model
	elif  framework == 'bigdl':
		model_json = keras_model.to_json()
		path = "model.json"
		with open(path, "w") as json_file:
			json_file.write(model_json)
		from bigdl.nn.layer import *
		bigdl_model = Model.load_keras(json_path=path)
		return bigdl_model
	else:
		raise ValueError('Unsupported framework:' + str(framework))

def tf_batch_generator(n_batches_for_epoch):
	input_shape=get_keras_input_shape(input_shapes[args.data])
	while 1:
		for i in range(n_batches_for_epoch):
			index_batch = range(X.shape[0])[batch_size*i:batch_size*(i+1)]       
			X_batch = X[index_batch,:].toarray().reshape((-1, input_shape[0], input_shape[1], input_shape[2]))
			y_batch = y[index_batch]
			yield (X_batch, np_utils.to_categorical(y_batch, num_labels[args.data]))

def get_framework_data(framework, X, y):
	input_shape=get_keras_input_shape(input_shapes[args.data])
	if framework == 'systemml':
		return X, y # Since SystemML supports Numpy, Scipy and Spark DF
	elif framework == 'tensorflow':
		if args.data_format == 'scipy':
			return None, None	
		elif args.data_format == 'numpy':
			print(str(input_shape))
			return X.reshape((-1, input_shape[0], input_shape[1], input_shape[2])), np_utils.to_categorical(y, num_labels[args.data])
		else:
			raise ValueError('Unsupported data format for tensorflow:' + args.data_format)
	elif framework == 'elephas':
		if args.data_format == 'numpy':
			# Rather than throwing an error that numpy and scipy is not supported:
			#from elephas.utils.rdd_utils import to_labeled_point
			#return to_labeled_point(sc, X, y, categorical=True), None
			from elephas.utils.rdd_utils import to_simple_rdd
			return to_simple_rdd(sc, X.reshape((-1, input_shape[0], input_shape[1], input_shape[2])), np_utils.to_categorical(y, num_labels[args.data])), None
		else:
			raise ValueError('TODO: will support alternative format for elephas once we get it to run with numpy')
	elif framework == 'bigdl':
		# BigDL requires the target to start from 1
		if args.data_format == 'numpy' or args.data_format == 'scipy':
			# Rather than throwing an error that numpy and scipy is not supported. Note: DF example is not working, so instead using RDD example
			rdd = sc.parallelize(range(y.shape[0])).map(lambda i : Sample.from_ndarray(X[0,:].toarray() if args.data_format == 'scipy' else X[0,:], y[0]))
			return rdd, None
		else:
			return X.rdd, y
	else:
		raise ValueError('Unsupported framework:' + str(framework))


framework = args.framework
t0 = time.time()
print("Getting model for the framework:" + framework)
framework_model = get_framework_model(framework)
t1 = time.time()
print("Getting data for the framework:" + framework)
framework_X, framework_y = get_framework_data(framework, X, y)
if args.data_format == 'spark_df':
	# For fair comparison with BigDL
	from pyspark import StorageLevel
	framework_X.persist(StorageLevel.MEMORY_AND_DISK)
	framework_X.count()
	#framework_X.show()
t2 = time.time()
data_loading = data_loading + t2 - t1
model_loading = t1 - t0 

if args.phase == 'train':
        print("Starting fit for the framework:" + framework)
        if framework == 'systemml':
		if args.data_format == 'scipy' or args.data_format == 'numpy':
                	framework_model.fit(framework_X, framework_y)
		elif args.data_format == 'spark_df':
			framework_model.fit(framework_X)
		else:
			raise ValueError('Unsupported data format for systemml:' + args.data_format)
        elif framework == 'tensorflow':
		if args.data_format == 'scipy':
			num_batches_per_epoch = int(math.ceil(X.shape[0]/batch_size))
			framework_model.fit_generator(generator=tf_batch_generator(num_batches_per_epoch), epochs=epochs, steps_per_epoch=num_batches_per_epoch)
                        K.clear_session()
		elif args.data_format == 'numpy':
                	framework_model.fit(framework_X, framework_y, epochs=epochs, batch_size=batch_size)
                	K.clear_session()
		else:
			raise ValueError('Unsupported data format for tensorflow:' + args.data_format)
        elif framework == 'elephas':
                framework_model.train(framework_X, nb_epoch=epochs, batch_size=batch_size)
        elif framework =='bigdl':
                from bigdl.examples.keras.keras_utils import *
                from bigdl.optim.optimizer import *
                from bigdl.util.common import *
                from bigdl.nn.criterion import *
                init_engine()
                #batch_size = 60 # make this divisible by number of cores
                #optim = SGD(learningrate=0.01, momentum=0.95, learningrate_decay=5e-4, nesterov=True, dampening=0) # Fails
                optim = Adam()
                bigdl_type = "float" if args.precision == 'single' else "double"
                optimizer = Optimizer(model=framework_model, training_rdd=framework_X, end_trigger=MaxEpoch(epochs), optim_method=optim, batch_size=batch_size, criterion=ClassNLLCriterion(logProbAsInput=False), bigdl_type=bigdl_type)
                optimizer.optimize()
        else:
                raise ValueError('Unsupported framework:' + str(framework))
t3 = time.time()
if args.phase == 'test':
	print("Starting predict for the framework:" + framework)
	if framework == 'systemml':
        	preds = framework_model.predict(framework_X)
		if hasattr(preds, '_jdf'):
			preds.count()
	elif framework == 'tensorflow':
		if args.data_format == 'scipy':
			num_batches_per_epoch = int(math.ceil(X.shape[0]/batch_size))
			framework_model.predict_generator(generator=batch_generator(num_batches_per_epoch), steps=num_batches_per_epoch)
                        K.clear_session()
		elif args.data_format == 'numpy':
        		framework_model.predict(framework_X)
	        	K.clear_session()
		else:
			raise ValueError('Unsupported data format for tensorflow:' + args.data_format)
	elif framework == 'elephas':
		if args.data_format != 'spark_df':
                        raise ValueError('Unsupported data format for elephas:' + args.data_format)
        	framework_model.predict(framework_X).count()
	elif framework =='bigdl':
		if args.data_format != 'spark_df':
                        raise ValueError('Unsupported data format for bigdl:' + args.data_format)
		from bigdl.util.common import *
        	init_engine()
	        #batch_size = 60 # make this divisible by number of cores
        	#optim = SGD(learningrate=0.01, momentum=0.95, learningrate_decay=5e-4, nesterov=True, dampening=0) # Fails
        	bigdl_type = "float" if args.precision == 'single' else "double"
		framework_model.predict(framework_X).count()
	else:
        	raise ValueError('Unsupported framework:' + str(framework))
end = time.time()
with open('time.txt', 'a') as f:
	f.write(args.framework + ',' + args.model + ',' + args.data + ',' + args.data_format + ',' + str(args.epochs) + ',' + str(args.batch_size) + ',' + str(args.num_gpus) + ',' + args.precision + ',' + args.blas + ',' + args.phase + ',' + args.codegen + ',' + str(end-start) + ',' + str(data_loading) + ',' + str(model_loading) + ',' + str(t3-t2) + ',' + str(spark_init_time) + ',' + str(end-t3) + '\n')

