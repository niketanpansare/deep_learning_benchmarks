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

DEFAULT_SPARK_PARAMS="--driver-memory 50g --conf spark.driver.maxResultSize=0"
SPARK_PARAMS=""
BIGDL_JAR="bigdl-SPARK_2.1-0.4.0-jar-with-dependencies.jar"
BIGDL_PARAMS="--driver-class-path "$BIGDL_JAR"  --conf spark.shuffle.reduceLocality.enabled=false --conf spark.shuffle.blockTransferService=nio --conf spark.scheduler.minRegisteredResourcesRatio=0.0 --conf spark.scheduler.minRegisteredResourcesRatio=1.0 --conf spark.speculation=false"

setup_env_framework() {
	if [ "$1" == 'bigdl' ]; then
		# BigDL doesnot support latest Keras version
		pip install keras==1.2.2
		#rm $BIGDL_JAR &> /dev/null
		#wget "http://central.maven.org/maven2/com/intel/analytics/bigdl/bigdl-SPARK_2.1/0.4.0/"$BIGDL_JAR
	else
		pip install keras==2.1.3
	fi
}

get_spark_params() {
	if [ "$1" == 'tensorflow' ]; then
		SPARK_PARAMS=$DEFAULT_SPARK_PARAMS
	elif [ "$1" == 'systemml' ]; then
		SPARK_PARAMS=$DEFAULT_SPARK_PARAMS
	elif [ "$1" == 'bigdl' ]; then
		SPARK_PARAMS=$DEFAULT_SPARK_PARAMS" "$BIGDL_PARAMS
	else
		echo "Unsupported framework:"$framework
                exit
	fi
}

get_framework_specific_settings() {
	declare -ga MODELS=("sentence_cnn_static" "lenet")
	if [ "$1" == 'tensorflow' ]; then
		declare -ga CODEGEN=("enabled" "disabled")
		declare -ga PRECISION=("single") # double precision fails on TF
		declare -ga NUM_GPUS=(0 1)
		declare -ga SUPPORTED_BLAS=("eigen")
	elif [ "$1" == 'systemml' ]; then
		declare -ga CODEGEN=("enabled" "disabled")
		declare -ga PRECISION=("single" "double")
		declare -ga NUM_GPUS=(0 1)
		declare -ga SUPPORTED_BLAS=("openblas") # ("none" "mkl" "openblas")
	elif [ "$1" == 'bigdl' ]; then
		declare -ga CODEGEN=("disabled")
		declare -ga PRECISION=("single" "double")
		declare -ga NUM_GPUS=(0)
		declare -ga SUPPORTED_BLAS=("mkl")
	else
		echo "Unsupported framework:"$framework
		exit
	fi
}

get_model_specific_settings() {
	framework=$1
	model=$2
	num_gpus=$3
	declare -ga EPOCHS=(1 5)
	declare -ga PHASES=("train") # test
	if [ "$framework" == 'tensorflow' ]; then
		declare -ga BATCH_SIZES=(60)
	elif [ "$framework" == 'systemml' ]; then
		declare -ga BATCH_SIZES=(60)
	elif [ "$framework" == 'bigdl' ]; then
                declare -ga BATCH_SIZES=(60) # BigDL only works if batch size is multiple of number of cores
        else
		echo "Unsupported framework:"$framework
                exit
        fi
	if [ "$model" == 'lenet' ]; then
                declare -ga DATA=("mnist")
		declare -ga DATA_FORMAT=("numpy") # since the dataset is tiny, no need to test with spark_df
        elif [ "$model" == 'sentence_cnn_static' ]; then
		# Sparse and large dataset
                declare -ga DATA=("imdb")
		if [ "$framework" == 'tensorflow' ]; then
                	declare -ga DATA_FORMAT=("scipy") # use generator as tensorflow doesnot support spark_df
		elif [ "$framework" == 'bigdl' ]; then
                        declare -ga DATA_FORMAT=("scipy") # bigdl+spark_df is crashing
		else
			declare -ga DATA_FORMAT=("spark_df")
		fi
        else
                echo "The model "$model" is not supported."
                exit
        fi
}

#if [[ -z "${SPARK_HOME}" ]]; then
#	echo "Please set the environment variable SPARK_HOME"
#else
	echo 'framework,model,data,data_format,epochs,batch_size,num_gpus,precision,blas,phase,codegen,total_time,data_loading_time,model_loading_time,fit_time,spark_init_time,predict_time' > time.txt
	if [ ! -d logs ]; then
		mkdir logs
	fi
	
	for framework in tensorflow systemml bigdl
	do
		pip uninstall keras --yes
		setup_env_framework "$framework"
		get_framework_specific_settings "$framework"
		get_spark_params "$framework"
		for model in "${MODELS[@]}"
		do
			for num_gpus in "${NUM_GPUS[@]}"
			do
				get_model_specific_settings $framework $model $num_gpus
				for epochs in "${EPOCHS[@]}"
				do
					for phase in "${PHASES[@]}"
					do
						for codegen in "${CODEGEN[@]}"
						do
							for precision in "${PRECISION[@]}"
							do
								for blas in "${SUPPORTED_BLAS[@]}"
								do
									for data in "${DATA[@]}"
									do
										for batch_size in "${BATCH_SIZES[@]}"
										do
											for data_format in "${DATA_FORMAT[@]}"
											do
												~/spark-2.1.0-bin-hadoop2.7/bin/spark-submit $SPARK_PARAMS compare_frameworks.py --model=$model --data=$data --data_format=$data_format --epochs=$epochs --batch_size=$batch_size --num_gpus=$num_gpus --framework=$framework --precision=$precision --blas=$blas --phase=$phase --codegen=$codegen &> logs/'log_'$model'_'$data'_'$epochs'_'$batch_size'_'$num_gpus'_'$framework'_'$precision'_'$blas'_'$phase'_'$data_format'_codegen-'$codegen'.txt'
											done
										done
									done
								done
							done
						done
					done
				done
			done
		done
	done
#fi
