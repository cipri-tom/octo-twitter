#! /usr/bin/env python3
"""
further helper files for the 2nd miniproject
"""

import numpy as np
import time, sys

def log_reading_time(func):
	""" Times the given function """
	def timed(*args, **kwargs):
		ts = time.time()
		result = func(*args, **kwargs)
		te = time.time()

		print(func.__name__, "Done. %3.2fs" % (te-ts))
		return result
	return timed


@log_reading_time
def load_data_and_labels(pos_path="../data/train_pos.txt", neg_path="../data/train_neg.txt", cut=None):
	"""
	Loads data from files, deletes repeated tweets and optionally removes long tweets to decrease zero padding and increase variance
	Returns sentences and labels.
	"""
	with open(neg_path, "r") as f:
		negative_examples = f.read().splitlines()

	# delete repeated tweets and ignore tweets with more than `cut` words
	num_tweets = {pos_path: 0, neg_path: 0}
	x_text = []
	for path in [pos_path, neg_path]:
		del_count, del_long_count = 0, 0
		prev_tweet = " "
		with open(path, "r") as file:
			for tweet in file:
				if tweet == prev_tweet:
					del_count += 1
					continue
				if not cut or tweet.count(" ") + 1 <= cut:  # avoid expensive split
					x_text.append(tweet)
					num_tweets[path] += 1
				elif cut:
					del_long_count += 1
				prev_tweet = tweet
		print("{}: #deleted repeated tweets:{}; #deleted long tweets:{}".format(path, del_count, del_long_count))

	# Generate labels
	positive_labels = [[0,1]] * num_tweets[pos_path]
	negative_labels = [[1,0]] * num_tweets[neg_path]
	y = np.concatenate([positive_labels, negative_labels], 0)

	return x_text, y

@log_reading_time
def load_test_data(test_data_file, max_document_length):
	""" Loads in test data, and splits into indices and tweets """
	ids, x_text = [], []
	with open(test_data_file, "r") as file:
		for line in file:
			id_, tweet = line.split(',', 1)
			ids.append(int(id_))
			assert(tweet.count(" ") + 1 <= max_document_length); "one of your test tweets is longer than allowed!"
			x_text.append(tweet)

	return ids, x_text

@log_reading_time
def vocab_processor(all_tweets):
	""" Builds vocab dictionary and transforms the sentences to array of word ids
	Note this is done from scratch every time, but pickled for use in prediction
	Replaces tensorflow.contrib.learn.preprocessing.VocabularyProcessor()
	"""
	all_tweets = [tweet.split(" ") for tweet in all_tweets]
	max_document_length = max({len(tweet) for tweet in all_tweets})
	print("Longest tweet: {} -> zero padding for the others".format(max_document_length))

	next_ID = 1  # giving unique ids for words
	d_wordIds = {}  # storing vectors
	x = np.zeros((len(all_tweets), max_document_length))
	for twt_idx, tweet in enumerate(all_tweets):
		for word_idx, word in enumerate(tweet):
			word_ID = d_wordIds.get(word, -1)  # do only one 'get'

			if word_ID == -1:
				word_ID = next_ID
				next_ID += 1
				d_wordIds[word] = word_ID

			x[twt_idx, word_idx] = word_ID

	import pickle
	# save vocab (we'll need it during the train)
	with open("../data/saved_vocab.pkl", 'wb') as f:
		pickle.dump(d_wordIds, f, pickle.HIGHEST_PROTOCOL)

	return d_wordIds, x

@log_reading_time
def map_test_data(x_text, max_document_length, saved_vocab_file="../data/saved_vocab.pkl"):
	"""
	Replaces tensorflow.contrib.learn.preprocessing.VocabularyProcessor()
	with transforming the sentences to array of word ids (vocab dict is loaded from file...)
	"""

	import os
	import pickle

	assert (os.path.getsize(saved_vocab_file) != 0), "imported vocab file is empty"
	with open(saved_vocab_file, "rb") as f:
		d_wordIds = pickle.load(f)

	print("Loaded vocabulary with size: {}".format(len(d_wordIds)))
	print("Longest tweet: {} -> zero padding for the others\n".format(max_document_length))

	x = np.zeros((len(x_text), max_document_length))
	for i in range(len(x_text)):  # iterates over tweets
		for k, word in enumerate(x_text[i].split()):  # iterates over "words" in one tweet
			if word in d_wordIds:
				id_ = d_wordIds.get(word)
				x[i, k] = id_

	return x


@log_reading_time
def load_GloVe(GloVe="../data/embeddings.npy", vocab="../data/vocab_cut.txt"):
	"""
	Loads GloVe word vectors to a dictionary (easier to search for words later...)
	"""
	with open(vocab, "r") as f:
		words = f.read().splitlines() # only the words represented in GloVe (preprocessing drops some...)
	GloVe = np.load(GloVe)
	d_GloVe = {}
	for i, word in enumerate(words):
		d_GloVe[word] = GloVe[i,:]

	return d_GloVe


@log_reading_time
def initW_embedding_GloVe(d_wordIds, embedding_dim,
						  GloVe_path="../data/embeddings.npy",
						  vocab="../data/vocab_cut.txt"):
	"""
	Builds weight matrix for embeddig layer (based on GloVe trained on the training tweets)
	"""
	d_GloVe = load_GloVe(GloVe_path, vocab)

	assert (d_GloVe.popitem()[1].shape[0] == embedding_dim), "embedding_dim flag and GloVe dim doesn't match!"

	initW = np.random.uniform(-1, 1, (len(d_wordIds)+1, embedding_dim))
	initW[0, :] = np.zeros((1, embedding_dim))  # wordID = 0 -> zero padded words
	for word, id_ in d_wordIds.items():
		# check if it's represented as GloVe vector:
		if word in d_GloVe:
			initW[id_, :] = d_GloVe.get(word).reshape(1,embedding_dim)

	return initW


@log_reading_time
def initW_embedding_pretrainedGloVe(d_wordIds, pretrainedGloVe, embedding_dim):
	"""
	Builds weight matrix for embeddig layer (based on pretrained GloVe)
	! 1st download pretrained (on Twitter dataset) vectors: http://nlp.stanford.edu/projects/glove/
	"""
	# open and sanity check:
	f = open(pretrainedGloVe, "r")
	num_dimensions = len(f.readline().split()) - 1 # first one is the word
	assert(num_dimensions == embedding_dim)
	f.seek(0)  # reset file

	initW = np.random.uniform(-1, 1, (len(d_wordIds)+1, embedding_dim))
	initW[0, :] = np.zeros((1, embedding_dim))  # wordID = 0 -> zero padded words
	remaining_words = len(d_wordIds)
	for line in f:
		split_line = line.split()
		word = split_line[0]
		if word in d_wordIds:
			id_ = d_wordIds.get(word)
			# embedding = [float(val) for val in split_line[1:]]
			initW[id_, :] = split_line[1:]
			remaining_words -= 1
		if remaining_words == 0:  # stop when we found all the words present in our dataset
			break
	f.close()

	return initW

@log_reading_time
def initW_embedding_pretrained_word2vec(d_wordIds, pretrained_word2vec, embedding_dim):
	"""
	Builds weight matrix for embeddig layer (based on pretrained GloVe)
	! 1st download pretrained vectors: https://code.google.com/archive/p/word2vec/
	"""

	assert (embedding_dim == 300), "embedding_dim flag and word2vec dim (300) doesn't match!"

	from gensim.models import Word2Vec as w2v

	word2vec = w2v.load_word2vec_format(pretrained_word2vec, binary=True)  # -> loads in the whole file ~ 4 GB RAM (iterating over the file is more than 8GB RAM)
	initW = np.random.uniform(-1, 1, (len(d_wordIds)+1, embedding_dim))
	initW[0, :] = np.zeros((1, embedding_dim))  # wordID = 0 -> zero padded words
	for word, id_ in d_wordIds.items():
		if word in word2vec:
			initW[id_, :] = word2vec[word]

	return initW


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	adapted from: https://github.com/dennybritz/cnn-text-classification-tf
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(len(data)/batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

