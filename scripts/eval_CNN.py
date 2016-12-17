#! /usr/bin/env python
"""
predicting with trained CNN based on: https://github.com/dennybritz/cnn-text-classification-tf
slightly modified to use the dataset of the 2nd miniproject
"""

import os
import csv
import time
import pickle
import datetime
import helpers
import numpy as np
import tensorflow as tf
from CNN import TextCNN

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", "runs/1482008771/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("max_document_length", 128, "max doc length during training (default: 64)")  # zero pedding for the shorter ones...
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
print("Loading data ...")
ids, x_text = helpers.load_test_data()

# Build vocabulary
x_test = helpers.map_test_data(x_text, FLAGS.max_document_length)

print("Evaluating {} inputs ...\n".format(len(ids)))

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# convert 0s to -1s
ids_ = np.where(all_predictions == 0)[0]
all_predictions[ids_] = -1

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(ids), all_predictions.astype(np.int32)))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(f, delimiter=",", fieldnames=fieldnames)
    writer.writeheader()
    csv.writer(f).writerows(predictions_human_readable)

