<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->

The tests in this directory aim to compare the prediction performance of TensorFlow and SystemML 
in the best-case scenario. 

## SystemML setup
- We ignore Py4J overhead in transferring the input batch.
- However, to make sure that we are forcing execution of every iteration, we print the sum of the output probability in the every iteration.
- We force SystemML to load the weights from filesystem.

## Tensorflow setup
- Here we generate the input matrix once, and invoke Keras' predict for every iteration. 
- After every iteration we print '.' to track the progress.
- Unlike SystemML, we donot perform an additional 'sum' operation on the output probabilities.
- The weights are initialized in-memory using Keras' API and passed directly to TensorFlow.
