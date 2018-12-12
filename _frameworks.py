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

import _models
import math, os
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, MaxPooling2D, Dropout, Flatten, Reshape
from keras import backend as K
from keras.models import Model
from keras.utils import np_utils
from keras.backend.tensorflow_backend import set_session
from keras import regularizers
from keras.preprocessing import sequence
import tensorflow as tf

def disable_gpu():
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = ''

def enable_or_disable_gpu(args):
	if args.num_gpus == 0:
		disable_gpu()
	elif args.num_gpus == 1:
		os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
		os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def set_visible_gpus(args):
	if args.framework == 'systemml':
		# When framework is systemml, force any tensorflow allocation to happen on CPU to avoid GPU OOM
		disable_gpu()
	elif args.framework == 'tensorflow':
		enable_or_disable_gpu(args)

def initialize_spark(args):
	try:
		spark = None
		from pyspark import SparkContext
		from pyspark.mllib.util import MLUtils
		sc = SparkContext()
		sc.parallelize([1, 2, 3, 4, 5]).count()
		if args.framework != 'elephas':
			# As elephas only support Spark 1.6 or less as of Feb 5th 2018
			from pyspark.sql import SparkSession
			spark = SparkSession.builder.getOrCreate()
			spark.createDataFrame([(1, 4), (2, 5), (3, 6)], ["A", "B"]).count()
		return sc, spark
	except ImportError:
        	raise ImportError('Make sure you are running with Spark. Hint: $SPARK_HOME/bin/spark-submit compare_framework.py -h')


def init_session(args):
	if args.framework == 'tensorflow':
		tf_config = tf.ConfigProto()
		#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
		if args.codegen == 'enabled':
			tf_config.graph_options.optimizer_options.global_jit_level = K.tf.OptimizerOptions.ON_1
		else:
			tf_config.graph_options.optimizer_options.global_jit_level = 0
		set_session(tf.Session(config=tf_config))		
		if args.precision == 'double':
			K.set_floatx('float64')
			print('WARNING: double precision is not supported in Keras. See https://stackoverflow.com/questions/48552508/running-keras-with-double-precision-fails/')

def compile_keras_model(args, keras_model):
	loss = 'categorical_crossentropy' if args.data =='mnist' else 'mean_squared_error'
	keras_model.compile(loss=loss, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.95, decay=5e-4, nesterov=True))
	return keras_model

def get_tensorflow_model(args, keras_model):
	if args.num_gpus >= 2:
		from keras.utils import multi_gpu_model
		keras_model = multi_gpu_model(keras_model, gpus=args.num_gpus)
	return compile_keras_model(args, keras_model)

def get_systemml_model(args, keras_model):
	from systemml.mllearn import Keras2DML
	load_keras_weights = True if args.phase == 'test' else False
	if args.num_samples  <= 0:
		raise ValueError("Incorrect number of samples")
	else:
		max_iter = int(args.epochs*math.ceil(args.num_samples/args.batch_size))
        sysml_model = Keras2DML(args.spark, compile_keras_model(args, keras_model), input_shape=args.input_shapes, batch_size=args.batch_size, max_iter=max_iter, test_iter=0, display=100, load_keras_weights=load_keras_weights)
	sysml_model.setStatistics(True).setStatisticsMaxHeavyHitters(100)
	sysml_model.setConfigProperty("sysml.gpu.sync.postProcess", "false")
	# sysml_model.setConfigProperty("sysml.stats.finegrained", "true")  # for detailed statistics
	# sysml_model.setConfigProperty("sysml.gpu.eager.cudaFree", "true") # conservative memory strategy for GPUs
	sysml_model.setConfigProperty("sysml.native.blas", args.blas)
	sysml_model.setConfigProperty("sysml.floating.point.precision", args.precision)
	if args.codegen == 'enabled':
		sysml_model.setConfigProperty("sysml.codegen.enabled", "true")
	if args.num_gpus >= 1:
		sysml_model.setGPU(True).setForceGPU(True)
	if args.num_gpus > 1:
		sysml_model.set(train_algo="allreduce_parallel_batches", parallel_batches=args.num_gpus)
	return sysml_model

def get_elephas_model(args, keras_model):
	from elephas.spark_model import SparkModel
	from elephas import optimizers as elephas_optimizers
	optim = elephas_optimizers.SGD(lr=0.01, momentum=0.95, decay=5e-4, nesterov=True)
	spark_model = SparkModel(args.sc, compile_keras_model(args, keras_model), optimizer=optim,mode='synchronous') #, num_workers=2)
	return spark_model

def get_bigdl_model(args, keras_model):
	model_json = compile_keras_model(args, keras_model).to_json()
	path = "model.json"
	with open(path, "w") as json_file:
		json_file.write(model_json)
	from bigdl.nn.layer import *
	bigdl_model = Model.load_keras(json_path=path)
	return bigdl_model

FRAMEWORK_MODELS = {'tensorflow':get_tensorflow_model, 'systemml':get_systemml_model, 'elephas':get_elephas_model, 'bigdl':get_bigdl_model}

def get_tensorflow_data(args, X, y):
	input_shape = _models.get_keras_input_shape(args.input_shapes)
	num_labels = args.num_labels if args.data == 'random' else _models.num_labels[args.data]
	if args.data_format == 'scipy':
		return X, y
	elif args.data_format == 'numpy':
		return X.reshape((-1, input_shape[0], input_shape[1], input_shape[2])), np_utils.to_categorical(y, num_labels)
	else:
		raise ValueError('Unsupported data format for tensorflow:' + args.data_format)

def get_systemml_data(args, X, y):
	return X, y

def get_elephas_data(args, X, y):
	input_shape = _models.get_keras_input_shape(args.input_shapes)
	num_labels = args.num_labels if args.data == 'random' else _models.num_labels[args.data]
	if args.data_format == 'numpy':
		# Rather than throwing an error that numpy and scipy is not supported:
		#from elephas.utils.rdd_utils import to_labeled_point
		#return to_labeled_point(sc, X, y, categorical=True), None
		from elephas.utils.rdd_utils import to_simple_rdd
		return to_simple_rdd(args.sc, X.reshape((-1, input_shape[0], input_shape[1], input_shape[2])), np_utils.to_categorical(y, num_labels)), None
	else:
		raise ValueError('TODO: will support alternative format for elephas once we get it to run with numpy')

def get_bigdl_data(args, X, y):
	from bigdl.util.common import Sample, init_engine
	from bigdl.optim.optimizer import Adam, MaxEpoch,Optimizer
	from bigdl.nn.criterion import ClassNLLCriterion
	# BigDL requires the target to start from 1
	if args.data_format == 'numpy' or args.data_format == 'scipy':
		# Rather than throwing an error that numpy and scipy is not supported. Note: DF example is not working, so instead using RDD example
		data_format = args.data_format
		rdd = args.sc.parallelize(range(y.shape[0])).map(lambda i : Sample.from_ndarray(X[0,:].toarray() if data_format == 'scipy' else X[0,:], y[0]))
		return rdd, None
	else:
		return X.rdd, y

FRAMEWORK_DATA = {'tensorflow':get_tensorflow_data, 'systemml':get_systemml_data, 'elephas':get_elephas_data, 'bigdl':get_bigdl_data}

def systemml_fit(args, framework_model, X, y):
	if args.data_format == 'scipy' or args.data_format == 'numpy' or args.data_format == 'binary_blocks':
		framework_model.fit(X, y)
	elif args.data_format == 'spark_df':
		framework_model.fit(X)
	else:
		raise ValueError('Unsupported data format for systemml:' + args.data_format)

def tf_batch_generator(n_batches_for_epoch, X, y):
        input_shape = _models.get_keras_input_shape(args.input_shapes)
        while 1:
                for i in range(n_batches_for_epoch):
                        index_batch = range(X.shape[0])[batch_size*i:batch_size*(i+1)]
                        X_batch = X[index_batch,:].toarray().reshape((-1, input_shape[0], input_shape[1], input_shape[2]))
                        y_batch = y[index_batch]
                        yield (X_batch, np_utils.to_categorical(y_batch, _models.num_labels[args.data]))

def tensorflow_fit(args, framework_model, X, y):
	if args.data_format == 'scipy':
		num_batches_per_epoch = int(math.ceil(X.shape[0]/args.batch_size))
		framework_model.fit_generator(generator=tf_batch_generator(num_batches_per_epoch, X, y), epochs=args.epochs, steps_per_epoch=num_batches_per_epoch)
	elif args.data_format == 'numpy':
		framework_model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size)
	else:
		raise ValueError('Unsupported data format for tensorflow:' + args.data_format)
	K.clear_session()

def elephas_fit(args, framework_model, X, y):
	framework_model.train(X, nb_epoch=args.epochs, batch_size=args.batch_size)

def bigdl_fit(args, framework_model, X, y):
	bigdl_type = "float" if args.precision == 'single' else "double"
	init_engine(bigdl_type)
	# SGD(learningrate=0.01, momentum=0.95, learningrate_decay=5e-4, nesterov=True, dampening=0) fails and
	# Make batch_size divisible by number of cores
	optim = Adam()
	optimizer = Optimizer(model=framework_model, training_rdd=X, end_trigger=MaxEpoch(args.epochs), optim_method=optim, batch_size=args.batch_size, criterion=ClassNLLCriterion(logProbAsInput=False), bigdl_type=bigdl_type)
	optimizer.optimize()

FRAMEWORK_FIT = {'systemml': systemml_fit, 'tensorflow': tensorflow_fit, 'elephas': elephas_fit, 'bigdl': bigdl_fit }


def systemml_predict(args, framework_model, X):
	preds = framework_model.predict(X)
	if hasattr(preds, '_jdf'):
		preds.count()

def tensorflow_predict(args, framework_model, X):
	if args.data_format == 'scipy':
		raise ValueError('Not implemented')
	elif args.data_format == 'numpy':
		framework_model.predict(X)
	else:
		raise ValueError('Unsupported data format for tensorflow:' + args.data_format)
	K.clear_session()

def elephas_predict(args, framework_model, X):
	if args.data_format != 'spark_df':
		raise ValueError('Unsupported data format for elephas:' + args.data_format)
	framework_model.predict(X).count()

def bigdl_predict(args, framework_model, X):
	if args.data_format != 'spark_df':
		raise ValueError('Unsupported data format for bigdl:' + args.data_format)
	bigdl_type = "float" if args.precision == 'single' else "double"
	init_engine(bigdl_type)
	framework_model.predict(X).count()

FRAMEWORK_PREDICT = {'systemml': systemml_predict, 'tensorflow': tensorflow_predict, 'elephas': elephas_predict, 'bigdl': bigdl_predict }
