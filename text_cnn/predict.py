#!/usr/bin/python
# coding=utf-8

import tensorflow as tf
import numpy as np
import os
import csv

from text_cnn import TextCNN
from SST_data import SSTDataset
from tensorflow.contrib import learn

# Data parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Positive data")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Negative data")

# Eval parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{attr}={value}".format(attr.upper(), value))
print("")

# Map data to vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restart(vocab_path)

if FLAGS.eval_train:
    #dataset = SSTDataset(FLAGS.positive_data_file, FLAGS.negative_data_file, vocab_processor=vocab_processor)
    pass
else:
    raw_x = ["a masterpiece four years in the making", "everything is off."]
    x = vocab_processor.transform(raw_x)
    y = [1, 0]

print("\nEvaluatin...\n")

# Evaluation
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        preds = sess.run(predictions, {input_x:x, dropout_keep_prob:1.0})

if y is not None:
    correct_predictions = float(sum(preds == y))
    print("Total number of test exmpales: {}".format(len(y)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y))))

predictions_human_readable = np.column_stack((np.array(raw_x), preds))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)