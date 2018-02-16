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
import math
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

# Model/Data-specific constants
sentence_length, embedding_dim = 2494, 300 # Maximum sentence length in imdb is 2494
input_shapes = { 'mnist':(1,28,28), 'imdb':(1, sentence_length, embedding_dim) }
num_labels = {'mnist':10, 'imdb':2 }

try:
        import tensorflow as tf
except ImportError:
        raise ImportError('Make sure you have installed tensorflow')

def validate_keras_version(args):
	right_keras_version = '1.2.2' if args.framework == 'bigdl' else '2.1.3'
	if keras.__version__ != right_keras_version:
		print('WARNING: Incorrect keras version:' + str(keras.__version__) + '. Expected keras version:' + right_keras_version)

def get_keras_input_shape(input_shape):
        #if K.image_data_format() == 'channels_first':
        #        return input_shape
        #else:
        return ( input_shape[1], input_shape[2], input_shape[0] )

def conv2d(args, nb_filter, nb_row, nb_col, activation='relu', padding='same', input_shape=None):
	if args.framework == 'bigdl':
                # bigdl only supports keras 1.2
                from keras.layers import Convolution2D
        else:
                from keras.layers import Conv2D
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

def dense(args, num_out, activation='relu'):
        if args.framework == 'bigdl':
                return Dense(num_out, activation=activation, W_regularizer=regularizers.l2(0.01))
        else:
                return Dense(num_out, activation=activation, kernel_regularizer=regularizers.l2(0.01))

def concat(args, conv_ngrams):
	if args.framework == 'bigdl':
                # bigdl only supports keras 1.2
                from keras.layers import merge
        else:
                from keras.layers import Concatenate
        if len(conv_ngrams) == 1:
                return conv_ngrams[0]
        if args.framework == 'bigdl':
                return merge(conv_ngrams, mode='concat', concat_axis=1)
        else:
                return Concatenate()(conv_ngrams)

def reshape_for_bigdl(args, inputL=None):
        input_shape = get_keras_input_shape(input_shapes[args.data]) # if args.framework == 'tensorflow' else input_shapes[args.data]
        # As per https://github.com/intel-analytics/BigDL/blob/master/pyspark/bigdl/models/lenet/lenet5.py
        if inputL == None:
                return Reshape(input_shape, input_shape=(1, input_shape[0]*input_shape[1]*input_shape[2]))
        else:
                return Reshape(input_shape)(inputL)

def get_lenet(args):
	validate_keras_version(args)
	keras_model = Sequential()
	if args.framework == 'bigdl':
		keras_model.add(reshape_for_bigdl(args))
	keras_model.add(conv2d(args, 32, 5, 5, input_shape=get_keras_input_shape(input_shapes[args.data])))
	keras_model.add(MaxPooling2D(pool_size=(2, 2)))
	keras_model.add(conv2d(args, 64, 5, 5))
	keras_model.add(MaxPooling2D(pool_size=(2, 2)))
	keras_model.add(Flatten())
	keras_model.add(dense(args, 512))
	keras_model.add(Dropout(0.5))
	keras_model.add(dense(args, 10, activation='softmax'))
	return keras_model

def get_sentence_cnn_static(args):
	validate_keras_version(args)
	conv_ngrams = []
	nb_filters = 100
	inputL = Input(shape=get_keras_input_shape(input_shapes[args.data]))
	inputL1 = reshape_for_bigdl(args, inputL) if args.framework == 'bigdl' else inputL
	kernel_h = [2, 3, 4, 5]
	pool_h = [ sentence_length - k for k in kernel_h ]
	for i in range(len(kernel_h)):
		conv = conv2d(args, nb_filters, kernel_h[i], embedding_dim, activation='relu', padding='valid')(inputL1)
		conv = MaxPooling2D(pool_size=(pool_h[i], 1))(conv)
		conv = Flatten()(conv)
		conv_ngrams.append(conv)
	z = concat(conv_ngrams)
	z = dense(args, 500, activation='relu')(z)
	z = Dropout(0.5)(z)
	model_output = dense(args, 2, activation='sigmoid')(z)
	keras_model = Model(inputL, model_output)
	return keras_model

MODELS={'lenet': get_lenet, 'sentence_cnn_static': get_sentence_cnn_static}
