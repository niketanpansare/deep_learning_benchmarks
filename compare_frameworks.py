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
parser.add_argument('--model', help='The model to use for comparison. Default: lenet', type=str, default='lenet', choices=['lenet', 'sentence_cnn_static'])
parser.add_argument('--data', help='The dataset to use for training/testing. Default: mnist', type=str, default='mnist', choices=['mnist', 'imdb'])
parser.add_argument('--data_format', help='The input format to use for reading the dataset. Default: numpy', type=str, default='numpy', choices=['spark_df', 'numpy', 'scipy'])
parser.add_argument('--epochs', help='Number of epochs. Default: 10', type=int, default=10)
parser.add_argument('--batch_size', help='Batch size. Default: 64', type=int, default=64)
parser.add_argument('--num_gpus', help='Number of GPUs. Default: 0', type=int, default=0)
parser.add_argument('--framework', help='The framework to use for running the benchmark. Default: systemml', type=str, default='systemml', choices=['systemml', 'tensorflow', 'bigdl'])
parser.add_argument('--precision', help='Floating point precision. Default: single', type=str, default='single', choices=['single', 'double'])
parser.add_argument('--blas', help='Native BLAS. Default: openblas', type=str, default='openblas', choices=['openblas', 'mkl', 'none', 'eigen'])
parser.add_argument('--phase', help='Training/testing phase. Default: train', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--codegen', help='Whether to apply codegen optimization. Supported values are: enabled, disabled. Default: disabled', type=str, default='disabled', choices=['enabled', 'disabled'])
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
input_shapes = _models.input_shapes[args.data]
args.input_shapes = input_shapes
num_features = input_shapes[0]*input_shapes[1]*input_shapes[2]
import keras
from keras import backend as K
_frameworks.init_session(args)

def load_scipy(args, input_file_path):
	if not os.path.exists(input_file_path):
		raise ValueError('Please generate the input file ' + input_file_path + ' before invoking compare_frameworks.')
	X, y = load_svmlight_file(input_file_path, n_features=num_features, zero_based=False)
	y = y if args.framework == 'bigdl' else y-1 # convert one-based labels to zero-based
	return X, y
def load_numpy(args, input_file_path):
	X, y = load_scipy(args, input_file_path)
	X = X.toarray().astype(np.float64) if args.precision == 'double' else X.toarray()
	return X, y
def load_spark_df(args, input_file_path):
	from pyspark.mllib.util import MLUtils
	X = MLUtils.convertVectorColumnsToML(MLUtils.loadLibSVMFile(sc, input_file_path, multiclass=False if args.data == 'imdb' else True, numFeatures=num_features).toDF())
	return X, None
DATA_LOADERS = {'scipy': load_scipy, 'numpy': load_numpy, 'spark_df': load_spark_df}

with measure('data_loading'):
	X, y = DATA_LOADERS[args.data_format](args, args.data + '.libsvm')
	framework_X, framework_y = _frameworks.FRAMEWORK_DATA[args.framework](args, X, y)
	args.num_samples = X.shape[0] if hasattr(X, 'shape') else -1
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

