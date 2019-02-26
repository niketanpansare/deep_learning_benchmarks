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

# To check usage, run with -h option: `python generate_data.py -h`

import os, argparse, sys, random
import numpy as np
from keras.preprocessing import sequence
from keras.datasets import imdb
from sklearn.datasets import dump_svmlight_file
from sklearn.utils import shuffle

parser=argparse.ArgumentParser()
parser.add_argument('--data', help='Supported values are: mnist, imdb, random. Default: mnist', type=str, default='mnist')
parser.add_argument('--num_samples', help='Number of samples to use. Default:-1 (implies use all the samples in the dataset', type=int, default=-1)
parser.add_argument('--num_features', help='Generate random data of the given number of features. Default: -1', type=int, default=-1)
parser.add_argument('--num_labels', help='Generate random data of the given number of labels. Default: -1', type=int, default=-1)
args=parser.parse_args()

if args.data == 'random':
	if args.num_samples <= 0 or args.num_features <= 0 or args.num_labels <= 0:
		raise ValueError('Number of samples (' + str(args.num_samples) + '), features (' + str(args.num_features) + ') and labels (' + str(args.num_labels) + ') should be greater than zero for --data=' + args.data)
elif args.num_features > 0 or args.num_labels > 0:
	print('Number of features (' + str(args.num_features) + ') and labels (' + str(args.num_labels) + ') are ignored for --data=' + args.data)

if args.data == 'imdb':
	sentence_length, embedding_dim = 2494, 300 # Maximum sentence length in imdb is 2494
	(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000) # load top-10K words
	X = np.hstack((x_train, x_test))
	y = np.hstack((y_train, y_test))
	X = sequence.pad_sequences(X, maxlen=sentence_length, padding="post", truncating="post")
	vocabulary = imdb.get_word_index()
	vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
	vocabulary_inv[0] = '<PAD/>'
	# since only sentence cnn static is implemented in this suite, use word2vec
	from gensim.models import word2vec
	if os.path.exists('imdb.libsvm'):
		embedding_model = word2vec.Word2Vec.load('imdb_embedding.model')
	else:
		embedding_model = word2vec.Word2Vec([[vocabulary_inv[w] for w in s] for s in X], size=embedding_dim, min_count=1, window=10, sample=1e-3)
		embedding_model.init_sims(replace=True)
		# Save the model in case the process crashes for reuse.
		embedding_model.save('imdb_embedding.model')
	def get_embedding(word):
		if word in embedding_model:
                        # one of the top-10K word.
                        return embedding_model[word]
		else:
			# Set unknown word and PAD to 0
			# Since pad is an artificial token, it can be replaced by 0's.
			# This will allow us to (1) test frameworks for sparsity support, and (2) reduce the size of generated libsvm file.
			return np.zeros(embedding_model.vector_size)
	embedding_weights = {key: get_embedding(word) for key, word in vocabulary_inv.items()}
	if os.path.exists('imdb.libsvm'):
		os.remove('imdb.libsvm')
	idx = range(y.shape[0])
	random.shuffle(idx)
	num_samples = args.num_samples if args.num_samples != -1 else len(y)
	idx = idx[0:num_samples]
	with open('imdb.libsvm', 'ab') as f:
		for i in idx:
			np_sentence = np.matrix(np.stack([embedding_weights[word] for word in X[i,:]]).flatten())
			dump_svmlight_file(np_sentence, [y[i]+1], f, zero_based=False)
elif args.data == 'mnist':
	import mnist
	# Use MNIST 60K dataset.
	X = mnist.train_images().reshape(60000, -1)
	y = mnist.train_labels()
	X_train, y_train =  shuffle(X, y)
	# Scale the input features
	scale = 0.00390625
	X_train = X_train*scale
	num_samples = args.num_samples if args.num_samples != -1 else len(y)
	dump_svmlight_file(X_train[0:num_samples,:], y_train[0:num_samples]+1, 'mnist.libsvm', zero_based=False)
elif args.data == 'random':
	X = np.random.rand(args.num_samples, args.num_features)
	y = np.random.randint(args.num_labels, size=args.num_samples)
	num_samples = args.num_samples
	dump_svmlight_file(X[0:num_samples,:], y[0:num_samples]+1, 'random.libsvm', zero_based=False)
else:
	raise ValueError('Unsupported data:' + args.data)
print("Done generating data.")
