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


import time, os, argparse, sys, math
from sklearn.datasets import load_svmlight_file
times = {}
start = time.time()
data_loading = 0.0

# The framework elephas (as of Feb 5th, 2018) fails with TypeError: 'LabeledPoint' object is not iterable
parser=argparse.ArgumentParser("Deep Learning (DL) Benchmarks.")
parser.add_argument('--model', help='The model to use for comparison. Default: lenet', type=str, default='lenet', choices=['lenet', 'sentence_cnn_static', 'vgg16', 'vgg19', 'resnet50'])
parser.add_argument('--data', help='The dataset to use for training/testing. Default: mnist', type=str, default='mnist', choices=['mnist', 'imdb', 'random'])
parser.add_argument('--data_format', help='The input format to use for reading the dataset. Default: numpy', type=str, default='numpy', choices=['spark_df', 'numpy', 'scipy', 'binary_blocks'])
parser.add_argument('--epochs', help='Number of epochs. Default: 10', type=int, default=10)
parser.add_argument('--batch_size', help='Batch size. Default: 64', type=int, default=64)
parser.add_argument('--num_gpus', help='Number of GPUs. Default: 0', type=int, default=0)
parser.add_argument('--num_channels', help='Number of channels when --data=random. Default: -1', type=int, default=-1)
parser.add_argument('--height', help='Image height when --data=random. Default: -1', type=int, default=-1)
parser.add_argument('--width', help='Image width when --data=random. Default: -1', type=int, default=-1)
parser.add_argument('--framework', help='The framework to use for running the benchmark. Default: systemml', type=str, default='systemml', choices=['systemml', 'tensorflow', 'bigdl'])
parser.add_argument('--precision', help='Floating point precision. Default: single', type=str, default='single', choices=['single', 'double'])
parser.add_argument('--blas', help='Native BLAS. Default: openblas', type=str, default='openblas', choices=['openblas', 'mkl', 'none', 'eigen'])
parser.add_argument('--phase', help='Training/testing phase. Default: train', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--codegen', help='Whether to apply codegen optimization. Supported values are: enabled, disabled. Default: disabled', type=str, default='disabled', choices=['enabled', 'disabled'])
parser.add_argument('--num_labels', help='Number of labels if --data==random. Default: -1', type=int, default=-1)
args=parser.parse_args()

from contextlib import contextmanager
@contextmanager
def measure(name):
    t1 = time.time()
    yield
    times[name] = str(time.time() - t1)

import _frameworks
_frameworks.set_visible_gpus(args)
with measure('spark_init_time'):
	args.sc, args.spark = _frameworks.initialize_spark(args)

import _models
if args.data == 'random':
	input_shapes = (args.num_channels, args.height, args.width)
	if args.num_labels <= 0:
		raise ValueError('The number of labels should be positive:' + str(args.num_labels))
else:
	input_shapes = _models.input_shapes[args.data]
args.input_shapes = input_shapes
num_features = input_shapes[0]*input_shapes[1]*input_shapes[2]
import keras
from keras import backend as K
_frameworks.init_session(args)

def load_scipy(args, input_file_path):
	if not os.path.exists(input_file_path):
		raise ValueError('Please generate the input file ' + input_file_path + ' before invoking compare_frameworks.')
	load_file_in_parts = True # To avoid https://github.com/scikit-learn/scikit-learn/issues/5269 when reading imdb dataset
	if load_file_in_parts:
		from scipy.sparse import vstack
		from itertools import islice
		X = None
		y = None
		with open(input_file_path) as f:
			part = list(islice(f, 1000))
			while len(part) > 0:
				with open(input_file_path + '_part', 'w') as f1:
					for line in part:
						f1.write(line)
				X_part, y_part = load_svmlight_file(input_file_path + '_part', n_features=num_features, zero_based=False)
				if X is None:
					X, y = X_part, y_part
				else:
					X, y = vstack([X, X_part]), vstack([y, y_part])
				part = list(islice(f, 1000))
	else:
		X, y = load_svmlight_file(input_file_path, n_features=num_features, zero_based=False)
	y = y if args.framework == 'bigdl' else y-1 # convert one-based labels to zero-based
	args.num_samples = X.shape[0]
	return X, y
def load_numpy(args, input_file_path):
	X, y = load_scipy(args, input_file_path)
	X = X.toarray().astype(np.float64) if args.precision == 'double' else X.toarray()
	args.num_samples = X.shape[0]
	return X, y
def load_spark_df(args, input_file_path):
	from pyspark.mllib.util import MLUtils
	X = MLUtils.convertVectorColumnsToML(MLUtils.loadLibSVMFile(args.sc, input_file_path, multiclass=False if args.data == 'imdb' else True, numFeatures=num_features).toDF())
	return X, None
def load_binary_blocks(args, input_file_path):
	df, ignore = load_spark_df(args, input_file_path)
	X_df = df.select("features")
	y_df = df.select("label")
	from systemml import MLContext, dml
	ml = MLContext(args.spark)
	res = ml.execute(dml('write(X, "X.mtx", format="binary"); write(y, "y.mtx", format="binary"); num_samples = nrow(X)').input(X=X_df, y=y_df).output("num_samples"))
	args.num_samples = res.get("num_samples")
	return "X.mtx", "y.mtx"
DATA_LOADERS = {'scipy': load_scipy, 'numpy': load_numpy, 'spark_df': load_spark_df, 'binary_blocks': load_binary_blocks}

with measure('data_loading'):
	X, y = DATA_LOADERS[args.data_format](args, args.data + '.libsvm')
	framework_X, framework_y = _frameworks.FRAMEWORK_DATA[args.framework](args, X, y)
	if args.data_format == 'spark_df':
        	# For fair comparison with BigDL
	        from pyspark import StorageLevel
        	framework_X.persist(StorageLevel.MEMORY_AND_DISK)
	        args.num_samples = framework_X.count()
with measure('model_loading'):
	framework_model = _frameworks.FRAMEWORK_MODELS[args.framework](args, _models.MODELS[args.model](args))
with measure('fit_time'):
	if args.phase == 'train':
		_frameworks.FRAMEWORK_FIT[args.framework](args, framework_model, framework_X, framework_y)
with measure('predict_time'):
	if args.phase == 'test':
		_frameworks.FRAMEWORK_PREDICT[args.framework](args, framework_model, framework_X)
end = time.time()
with open('time.txt', 'a') as f:
	f.write(args.framework + ',' + args.model + ',' + args.data + ',' + args.data_format + ',' + str(args.epochs) + ',' + str(args.batch_size) + ',' + str(args.num_gpus) + ',' + args.precision + ',' + args.blas + ',' + args.phase + ',' + args.codegen + ',' + str(end-start) + ',' + times['data_loading'] + ',' + times['model_loading'] + ',' + times['fit_time'] + ',' + times['spark_init_time'] + ',' + times['predict_time'] + '\n')

