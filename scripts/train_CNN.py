#! /usr/bin/env python3
"""
CNN training based on: https://github.com/dennybritz/cnn-text-classification-tf
slightly modified to load in pretrained word vectors (and to use the dataset of the 2nd miniproject)
"""

import os
import time
import datetime
import helpers
import numpy as np
import tensorflow as tf
from CNN import TextCNN


# Parameters
# ==================================================

# Model Hyperparameters

# use predefined embeddings
tf.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings (default: ../data/GoogleNews-vectors-negative300.bin)")
tf.flags.DEFINE_string("GloVe", "../data/glove.twitter.27B.200d.txt", "GloVe vectors with pre-trained embeddings (default: ../data/glove.twitter.27B.200d.txt)")
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")  # should be filled from file ...
tf.flags.DEFINE_string("pos_text", "../data/train-pos-full-preprocess.txt", "Path of text with positive examples (default: ../data/train_pos.txt)")
tf.flags.DEFINE_string("neg_text", "../data/train-neg-full-preprocess.txt", "Path of text with negative examples (default: ../data/train_neg.txt)")
tf.flags.DEFINE_integer("max_document_length", 60, "Cutting long tweets to decrease zero padding (default: None)")

tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate for Adam optimizer (default: 1e-3)")  # see more here: http://devdocs.io/tensorflow~python/train#AdamOptimizer
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")

tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("max_it", 500, "Terminate after this many steps (default: 2000)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y = helpers.load_data_and_labels(FLAGS.pos_text, FLAGS.neg_text, FLAGS.max_document_length)

# Build vocabulary
d_vocab, x = helpers.vocab_processor(x_text)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
x_train, x_dev = x_shuffled[:-2000], x_shuffled[-2000:]
y_train, y_dev = y_shuffled[:-2000], y_shuffled[-2000:]
print("Vocabulary Size: {:d}".format(len(d_vocab)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

import gc
x_text = []
y = []
x_shuffled = []
y_shuffled = []
gc.collect()

# Load word vectors


"""
Added by Andras to load in GloVe/word2vec vectors!
(before you start training steps you can assign W to whatever you want)
based on: https://github.com/dennybritz/cnn-text-classification-tf/issues/17
"""
# Don't try to use GloVe and word2vec in the same time
if FLAGS.GloVe:
    if FLAGS.GloVe == "../data/embeddings.npy":
        print("Load trained GloVe from {} ...\n".format(FLAGS.GloVe))
        initW = helpers.initW_embedding_GloVe(d_vocab, FLAGS.embedding_dim, FLAGS.GloVe, FLAGS.vocab_cut)
    elif FLAGS.GloVe in ["../data/glove.twitter.27B.25d.txt", "../data/glove.twitter.27B.50.txt", "../data/glove.twitter.27B.100d.txt", "../data/glove.twitter.27B.200d.txt"]:
        print("Load pretrained GloVe from {} ...\n".format(FLAGS.GloVe))
        initW = helpers.initW_embedding_pretrainedGloVe(d_vocab, FLAGS.GloVe, FLAGS.embedding_dim)
elif FLAGS.word2vec:
    print("Load trained word2vec from {} ...\n".format(FLAGS.word2vec))
    initW = helpers.initW_embedding_pretrained_word2vec(d_vocab, FLAGS.word2vec, FLAGS.embedding_dim)

##### +code added until here (see functions in the helpers file) #####

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=2,
            vocab_size=len(d_vocab),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            initW=initW,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            # train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """         
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs) 
            
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
            if current_step == FLAGS.max_it:
                break
