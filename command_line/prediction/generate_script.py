#!/usr/bin/python
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
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from pyspark.sql import SparkSession
from pyspark import SparkContext
from systemml.mllearn import Keras2DML

sc  = SparkContext()
spark = SparkSession.builder.getOrCreate()
sc.setLogLevel('ERROR')

import keras
keras.backend.set_image_data_format("channels_first")

import get_model
keras_model = get_model.keras_model
sysml_model = Keras2DML(spark, keras_model, weights='tmp_dir', max_iter=1, batch_size=64)
sysml_model.set(test_algo='batch', perform_one_hot_encoding=False)

num_iter = get_model.num_iter

with open('prediction.dml', 'w') as f:
	s = sysml_model.get_prediction_script()
	keras_shape = keras_model.layers[0].input_shape
	cols = 1
	for i in range(len(keras_shape)-1):
		cols = cols * keras_shape[i+1]
	# Generate random data
	s = s.replace('X_full = read(X_full_path)', 'X_full = rand(rows=64, cols=' + str(cols) + ')')
	if num_iter != 1:
		s = s.replace('Xb = X_full', 'for(i in 1:' + str(num_iter) + ') {\nXb = X_full\n')
	f.write(s)
	f.write('\nprint(sum(Prob))') # To make sure we execute the script
	if num_iter != 1:
		f.write('\n}')
