# Deep Learning Benchmarks

This benchmarks compares various single-node and distributed frameworks that 
- support either [Keras](https://keras.io/) and [Caffe](http://caffe.berkeleyvision.org/) format, 
- can be invoked via Python, 
- and use [Apache Spark](https://spark.apache.org/) for distributed operations. 

The first version will compare end-to-end workflow of popular deep learning models on real-world datasets (with varying characteristics) on different tasks. For layer-by-layer benchmarks, please see Soumith's [convnet benchmark](https://github.com/soumith/convnet-benchmarks/). We will only use **synchronous mini-batch SGD** to ignore tradeoffs between statistical and hardware efficiencies. 

I will do my best to ensure that these benchmarks compare systems fairly and in an unbiased manner. Please feel free to shoot me an email at {my github handle} @ gmail.com or create PR to point out any mistakes.

## Usage

### Generating the data

Please use `generate_data.py` script to generate the dataset before running the benchmark.

```
> $SPARK_HOME/bin/spark-submit generate_data.py -h

usage: generate_data.py [-h] [--data DATA] [--num_samples NUM_SAMPLES]
                        [--num_features NUM_FEATURES]
                        [--num_labels NUM_LABELS]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Supported values are: mnist, imdb, random. Default:
                        mnist
  --num_samples NUM_SAMPLES
                        Number of samples to use. Default:-1 (implies use all
                        the samples in the dataset
  --num_features NUM_FEATURES
                        Generate random data of the given number of features.
                        Default: -1
  --num_labels NUM_LABELS
                        Generate random data of the given number of labels.
                        Default: -1
```

### Executing the benchmark

You can either use `run.sh` script provided in the repository or invoke `compare_framework.py` to invoke individual tests.


```
> $SPARK_HOME/bin/spark-submit compare_frameworks.py -h

usage: Deep Learning (DL) Benchmarks. [-h]
                                      [--model {lenet,sentence_cnn_static,vgg16,vgg19,resnet50}]
                                      [--data {mnist,imdb,random}]
                                      [--data_format {spark_df,numpy,scipy,binary_blocks}]
                                      [--epochs EPOCHS]
                                      [--batch_size BATCH_SIZE] [--profile]
                                      [--num_gpus NUM_GPUS]
                                      [--num_channels NUM_CHANNELS]
                                      [--height HEIGHT] [--width WIDTH]
                                      [--framework {systemml,tensorflow,bigdl}]
                                      [--precision {single,double}]
                                      [--blas {openblas,mkl,none,eigen}]
                                      [--phase {train,test}]
                                      [--codegen {enabled,disabled}]
                                      [--num_labels NUM_LABELS]

optional arguments:
  -h, --help            show this help message and exit
  --model {lenet,sentence_cnn_static,vgg16,vgg19,resnet50}
                        The model to use for comparison. Default: lenet
  --data {mnist,imdb,random}
                        The dataset to use for training/testing. Default:
                        mnist
  --data_format {spark_df,numpy,scipy,binary_blocks}
                        The input format to use for reading the dataset.
                        Default: numpy
  --epochs EPOCHS       Number of epochs. Default: 1
  --batch_size BATCH_SIZE
                        Batch size. Default: 64
  --profile             Should profile. Default: False
  --num_gpus NUM_GPUS   Number of GPUs. Default: 0
  --num_channels NUM_CHANNELS
                        Number of channels when --data=random. Default: -1
  --height HEIGHT       Image height when --data=random. Default: -1
  --width WIDTH         Image width when --data=random. Default: -1
  --framework {systemml,tensorflow,bigdl}
                        The framework to use for running the benchmark.
                        Default: systemml
  --precision {single,double}
                        Floating point precision. Default: single
  --blas {openblas,mkl,none,eigen}
                        Native BLAS. Default: openblas
  --phase {train,test}  Training/testing phase. Default: train
  --codegen {enabled,disabled}
                        Whether to apply codegen optimization. Supported
                        values are: enabled, disabled. Default: disabled
  --num_labels NUM_LABELS
                        Number of labels if --data==random. Default: -1
```
