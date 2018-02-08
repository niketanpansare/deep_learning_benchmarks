# Deep Learning Benchmarks

This benchmarks compares various single-node and distributed frameworks that 
- support either [Keras](https://keras.io/) and [Caffe](http://caffe.berkeleyvision.org/) format, 
- can be invoked via Python, 
- and use [Apache Spark](https://spark.apache.org/) for distributed operations. 

The first version will compare end-to-end workflow of popular deep learning models on real-world datasets (with varying characteristics) on different tasks. For layer-by-layer benchmarks, please see Soumith's [convnet benchmark](https://github.com/soumith/convnet-benchmarks/). We will only use **synchronous mini-batch SGD** to ignore tradeoffs between statistical and hardware efficiencies. 

I will do my best to ensure that these benchmarks compare systems fairly and in an unbiased manner. Please feel free to shoot me an email at {my github handle} @ gmail.com or create PR to point out any mistakes.
