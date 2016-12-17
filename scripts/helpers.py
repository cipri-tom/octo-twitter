#! /usr/bin/env python3
"""
further helper files for the 2nd miniproject
"""

import numpy as np

def load_data_and_labels(positive_data_file="../data/train_pos.txt", negative_data_file="../data/train_neg.txt"):
	"""
	Loads data from files, and generates labels.
	Returns sentences and labels.
	"""
	# Load data from files
	with open(positive_data_file, "r") as f:
		positive_examples = f.read().splitlines()
	f.close()
	with open(negative_data_file, "r") as f:
		negative_examples = f.read().splitlines()
	f.close()
	x_text = positive_examples + negative_examples
	# Generate labels
	positive_labels = [[0,1] for _ in positive_examples]
	negative_labels = [[1,0] for _ in negative_examples]
	y = np.concatenate([positive_labels, negative_labels], 0)
    
	return x_text, y
	
	
def load_test_data(test_data_file="../data/test_data.txt"):
	"""
	loads in test data, and splits into indicies and tweets
	"""
	
	with open(test_data_file, "r") as f:
		tests = f.read().splitlines()
	f.close()
	
	ids = []
	x_text = []
	for tweet in tests:
		tmp = tweet.split(',', 1)
		ids.append(tmp[0])
		x_text.append(tmp[1])
		
	return ids, x_text 


def vocab_processor(x_text):
	"""
	Replaces tensorflow.contrib.learn.preprocessing.VocabularyProcessor()
	with buildig vocab dictionary and transforming the sentences to array of word ids
	"""

	max_document_length = max([len(x.split(" ")) for x in x_text])
	print("Longest tweet: {}, -> zero padding for the others".format(max_document_length))

	id_ = 1  # giving unique ids for words
	d_wordIds = {}  # storing vectors	
	x = np.zeros((len(x_text), max_document_length))
	for i in range(len(x_text)):  # iterates over tweets
		for k, word in enumerate(x_text[i].split()):  # iterates over "words" in one tweet
			if word not in d_wordIds:
				d_wordIds[word] = id_					
				x[i, k] = id_
				id_ += 1
			else:
				getId_ = d_wordIds.get(word)
				x[i, k] = getId_				
	
	#print("vocab size before pickle:", len(d_wordIds))
	import pickle
	# save vocab (we'll need it during the train)
	with open("../data/saved_vocab.pkl", 'wb') as f:
		pickle.dump(d_wordIds, f, pickle.HIGHEST_PROTOCOL)
	f.close()
								
	return d_wordIds, x
	

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
	f.close()
	print("Loaded vocabulary with size: {}".format(len(d_wordIds)))
	print("Longest tweet: {} -> zero padding for the others\n".format(max_document_length))
	
	x = np.zeros((len(x_text), max_document_length))
	for i in range(len(x_text)):  # iterates over tweets
		for k, word in enumerate(x_text[i].split()):  # iterates over "words" in one tweet
			if word in d_wordIds:
				id_ = d_wordIds.get(word)
				x[i, k] = id_
				
	return x
	

def load_GloVe(GloVe="../data/embeddings.npy", vocab="../data/vocab_cut.txt"):
	"""
	Loads GloVe word vectors to a dictionary (easier to search for words later...)
	"""	
	
	with open(vocab, "r") as f:
		words = f.read().splitlines() # only the words represented in GloVe (preprocessing drops some...)
	f.close()
	GloVe = np.load(GloVe)
	
	d_GloVe = {}
	for i, word in enumerate(words):
		d_GloVe[word] = GloVe[i,:]
	
	return d_GloVe


def initW_embedding_GloVe(d_wordIds, embedding_dim, GloVe="../data/embeddings.npy", vocab="../data/vocab_cut.txt"):
	"""
	builds weight matrix for embeddig layer (based on GloVe trained on the training tweets)
	"""
	
	d_GloVe = load_Glove()

	assert (d_GloVe.popitem()[1].shape[0] == embedding_dim), "embedding_dim flag and GloVe dim doesn't match!"
	
	initW = np.zeros((len(d_wordIds)+1, embedding_dim))
	for word, id_ in d_wordIds.items():
		# check if it's represented as GloVe vector:
		if word in d_GloVe:
			initW[id_, :] = d_GloVe.get(word).reshape(1,embedding_dim)
			
	return initW


def initW_embedding_pretrainedGloVe(d_wordIds, pretrainedGloVe, embedding_dim):
	"""
	builds weight matrix for embeddig layer (based on pretrained GloVe)
	! 1st download pretrained (on Twitter dataset) vectors: http://nlp.stanford.edu/projects/glove/
	"""
	
	word_count = len(d_wordIds)
	
	f = open(pretrainedGloVe, "r")
	
	initW = np.zeros((len(d_wordIds)+1, embedding_dim))
	i = word_count
	for line in f:
		split_line = line.split()
		word = split_line[0]
		if i == word_count:  # check dimensions (only once)
			embedding = [float(val) for val in split_line[1:]]
			assert (len(embedding) == embedding_dim), "embedding_dim flag and GloVe dim doesn't match!"
		if word in d_wordIds:
			id_ = d_wordIds.get(word)
			embedding = [float(val) for val in split_line[1:]]			
			initW[id_, :] = embedding
			i -= 1
		if i == 0:  # don't itrate more if we found all the words present in our dataset
			break
	
	f.close()
			
	return initW
	

def initW_embedding_pretrained_word2vec(d_wordIds, pretrained_word2vec, embedding_dim):
	"""
	builds weight matrix for embeddig layer (based on pretrained GloVe)
	! 1st download pretrained vectors: https://code.google.com/archive/p/word2vec/
	"""
	
	assert (embedding_dim == 300), "embedding_dim flag and word2vec dim (300) doesn't match!"
	
	from gensim.models import Word2Vec as w2v

	word2vec = w2v.load_word2vec_format(pretrained_word2vec, binary=True)  # -> loads in the whole file ~ 4 GB RAM (iterating over the file is more than 8GB RAM)
	initW = np.zeros((len(d_wordIds)+1, embedding_dim))
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

