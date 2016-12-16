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


def load_GloVe(vectors, vocab="../data/vocab_cut.txt"):
	"""
	Loads GloVe word vectors to a dictionary (easier to search for words later...)
	"""	
	
	with open(vocab, "r") as f:
		words = f.read().splitlines() # only the words represented in GloVe (preprocessing drops some...)
	f.close()
	GloVe = np.load(vectors)
	
	d_GloVe = {}
	for i, word in enumerate(words):
		d_GloVe[word] = GloVe[i,:]
	
	return d_GloVe


def vocab_processor(x_text):
	"""
	Replaces tensorflow.contrib.learn.preprocessing.VocabularyProcessor()
	with buildig vocab dictionary and transforming the sentences to array of word ids
	"""
	max_document_length = max([len(x.split(" ")) for x in x_text])

	id_ = 1  # giving unique ids for words
	d_wordIds = {}  # storing vectors	
	x = np.zeros((len(x_text), max_document_length))
	for i in range(len(x_text)):  # iterates over tweets
		for k, word in enumerate(x_text[i].split()):  # iterates over "words" in one tweet
			if word not in d_wordIds:
				d_wordIds[word] = id_					
				x[i, k] = id_
			else:
				getId_ = d_wordIds.get(word)
				x[i, k] = getId_
								
	return d_wordIds, x

def init_embedding_W(d_wordIds, d_GloVe, embedding_dim):
	"""
	builds weight matrix for embeddig layer
	"""

	assert (d_GloVe.popitem()[1].shape[0] == embedding_dim), "embedding_dim flag and GloVe dim doesn't match!"
	
	initW = np.random.uniform(-1, 1,(len(d_wordIds), embedding_dim))  # randomly initialized words (the NN will learn...)
	for word, id_ in d_wordIds.items():
		# check if it's represented as GloVe vector:
		if word in d_GloVe:
			initW[id_, :] = d_GloVe.get(word).reshape(1,embedding_dim)
			
	return initW


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
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

