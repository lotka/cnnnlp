#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv,pickle

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("data_path","csv","Path to extracted books csv files")
tf.flags.DEFINE_float("test_sample_percentage",0.4,"Percentage to leave for testing")
tf.flags.DEFINE_integer('max_sentence_length',300,'Maximum length of sentences')
tf.flags.DEFINE_string("class_encoding","full","book review data has 5 classes, can either split into positive or negative or leave the all 5")
tf.flags.DEFINE_string("book_name",None,"book file name")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1488233333/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

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
# Load the csv
df = data_helpers.load_data(path=FLAGS.data_path,
                            max_sentence_length=FLAGS.max_sentence_length,
                            encoding=FLAGS.class_encoding,
                            book=FLAGS.book_name)
df = df.iloc[int(-FLAGS.test_sample_percentage*len(df)):]
x_test,y_test = data_helpers.dataframe_to_xy(df)
# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_test])

with open('embeddings.pkl','rb') as f:
    dictionary = pickle.load(f)['dictionary']

from tqdm import tqdm
x = []
for text in tqdm(x_test):
    split = text.split(' ')
    res = np.zeros(max_document_length)
    for i in xrange(min(len(split),max_document_length)):
        if split[i] in dictionary:
            res[i] =dictionary[split[i]]
        else:
            res[i] =0

    x.append(res)

x_test = np.array(x)

print("\nEvaluating...\n")

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
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in tqdm(batches):
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test.argmax(axis=1)))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    print("Regression Accuracy: {:g}".format(np.sqrt(np.mean((all_predictions - y_test.argmax(axis=1))**2))))
