#!/bin/bash
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

SPARK_HOME=/home/npansar/spark-2.3.0-bin-hadoop2.7
$SPARK_HOME/bin/spark-submit --driver-memory 200g generate_script.py

START=$(date +%s)
$SPARK_HOME/bin/spark-submit --driver-memory 200g --jars systemml-1.3.0-SNAPSHOT-extra.jar  SystemML.jar -f prediction.dml -gpu -stats -nvargs weights=tmp_dir
END=$(date +%s)
DIFF1=$(python -c "print(${END} - ${START})")

START=$(date +%s)
$SPARK_HOME/bin/spark-submit --driver-memory 200g run_tf.py
END=$(date +%s)
DIFF2=$(python -c "print(${END} - ${START})")

echo "-------------------------------------------"
echo "Time: SystemML="$DIFF1", TensorFlow="$DIFF2
echo "-------------------------------------------"
