#!/usr/bin/python
# coding=utf-8

import tensorflow as tf
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import datetime

from text_cnn import TextCNN
from SST_data import SSTDataset

# Data Loading parameters
tf.flags.DEFINE_float("dev_validation_percentage", 0.1, "Percentage of validation data")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Positive data")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Negative data")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Word embedding dimension")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Filter sizes")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout ratio")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization labmda")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_steps", 20000, "Number of steps")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate validation set every this steps")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")

# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{attr}={value}".format(attr=attr.upper(), value=value))
print("")


# Load datasets
dataset = SSTDataset(FLAGS.positive_data_file, FLAGS.negative_data_file, validation_perc=FLAGS.dev_validation_percentage)

# Training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=dataset.train_data.shape[1],
            num_classes=dataset.train_label.shape[1],
            vocab_size=len(dataset.vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda
        )

        # Training procedure

        # Summaries
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("writing to {dir}\n".format(dir=out_dir))

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        train_op = optimizer.minimize(cnn.loss, global_step)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Validation summaries
        val_summary_op = tf.summary.merge([loss_summary, acc_summary])
        val_summary_dir = os.path.join(out_dir, "summaries", "val")
        val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

        # Checkpoint directories
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        dataset.vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        for i in range(1, FLAGS.num_steps + 1):
            # Train
            current_step = tf.train.global_step(sess, global_step)
            batch_x, batch_y = dataset.next_batch(batch_size=FLAGS.batch_size, data_type="train")
            feed_dict = {
                cnn.input_x: batch_x,
                cnn.input_y: batch_y,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

            # Validate
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                feed_dict = {
                    cnn.input_x: dataset.val_data,
                    cnn.input_y: dataset.val_label,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, val_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict
                )
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if val_summary_writer:
                    val_summary_writer.add_summary(summaries, step)
            
            # Checkpoints
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to{}\n".format(path))
