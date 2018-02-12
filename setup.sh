#!/usr/bin/bash
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

# Install Keras:
read -r -p "Do you want to install Keras? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
	pip install --upgrade keras==2.1.3
fi

# Build tensorflow from the source with CUDA 8 and CuDNN 5 for fair comparison.
read -r -p "Do you want to compile TensorFlow 1.5 from source? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
	git clone https://github.com/tensorflow/tensorflow
	cd tensorflow
	git checkout r1.5
	echo "Please use CUDA 8, CuDNN 5 and enable XLA."
	./configure
	bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
	bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/
	pip install --upgrade ~/tensorflow-1.5.0-cp27-cp27mu-linux_x86_64.whl
fi

read -r -p "Do you want to install BigDL? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
	BIGDL_JAR="bigdl-SPARK_2.1-0.4.0-jar-with-dependencies.jar"
	wget "http://central.maven.org/maven2/com/intel/analytics/bigdl/bigdl-SPARK_2.1/0.4.0/"$BIGDL_JAR
	pip install --upgrade bigdl==0.4.0
fi

read -r -p "Do you want to install bleeding edge SystemML? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
	git clone https://github.com/apache/systemml.git
	cd systemml
	mvn package -P distribution
	pip install target/systemml-*-python.tar.gz
else
	read -r -p "Do you want to install released version of SystemML? [y/N] " response
	if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
	then
		pip install --upgrade systemml
	fi
fi

read -r -p "Do you want to install libraries necessary for generating the datasets? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
	pip install mnist gensim
fi


