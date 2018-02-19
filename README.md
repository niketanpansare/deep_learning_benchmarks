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

usage: generate_data.py [-h] [--data DATA]

optional arguments:
  -h, --help   show this help message and exit
  --data DATA  Supported values are: mnist, imdb. Default: mnist
```

### Executing the benchmark

You can either use `run.sh` script provided in the repository or invoke `compare_framework.py` to invoke individual tests.


```
> $SPARK_HOME/bin/spark-submit compare_framework.py -h

usage: Deep Learning (DL) Benchmarks. [-h]
                                      [--model {lenet,sentence_cnn_static}]
                                      [--data {mnist,imdb}]
                                      [--data_format {spark_df,numpy,scipy}]
                                      [--epochs EPOCHS]
                                      [--batch_size BATCH_SIZE]
                                      [--num_gpus NUM_GPUS]
                                      [--framework {systemml,tensorflow,bigdl}]
                                      [--precision {single,double}]
                                      [--blas {openblas,mkl,none,eigen}]
                                      [--phase {train,test}]
                                      [--codegen {enabled,disabled}]

optional arguments:
  -h, --help            show this help message and exit
  --model {lenet,sentence_cnn_static}
                        The model to use for comparison. Default: lenet
  --data {mnist,imdb}   The dataset to use for training/testing. Default:
                        mnist
  --data_format {spark_df,numpy,scipy}
                        The input format to use for reading the dataset.
                        Default: numpy
  --epochs EPOCHS       Number of epochs. Default: 10
  --batch_size BATCH_SIZE
                        Batch size. Default: 64
  --num_gpus NUM_GPUS   Number of GPUs. Default: 0
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
```


