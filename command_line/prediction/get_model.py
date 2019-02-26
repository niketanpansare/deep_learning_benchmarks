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

import keras
import numpy as np
# Initialized in generate_script
#keras.backend.set_image_data_format("channels_first")

num_iter = 100
batch_size = 64

def get_tensor(shape, random=True):
    if shape[0] is None:
        shape = list(shape)
        shape[0] = batch_size
    return (np.random.randint(100, size=shape) + 1) / 100
def initialize_weights(model):
    for l in range(len(model.layers)):
        if model.layers[l].get_weights() is not None or len(model.layers[l].get_weights()) > 0:
            model.layers[l].set_weights([get_tensor(elem.shape) for elem in model.layers[l].get_weights()])
    return model


keras_model = keras.applications.vgg16.VGG16()
keras_model = initialize_weights(keras_model)
keras_model.compile(optimizer='sgd', loss= 'categorical_crossentropy')
