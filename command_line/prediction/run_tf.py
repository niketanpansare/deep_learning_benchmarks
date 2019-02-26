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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np

from pyspark.sql import SparkSession
from pyspark import SparkContext
from systemml.mllearn import Keras2DML

sc  = SparkContext()
spark = SparkSession.builder.getOrCreate()
sc.setLogLevel('ERROR')

import get_model
keras_model = get_model.keras_model
keras_shape = keras_model.layers[0].input_shape
X = np.random.rand(64, keras_shape[1], keras_shape[2], keras_shape[3])
for i in range(get_model.num_iter):
	keras_model.predict(X)
	print('.')
