# coding=utf-8

import os
import argparse
import tensorflow as tf
from data import Dataset
import logging
logging.basicConfig(level=logging.DEBUG)

from model import Word2vec

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Word2vec')
  parser.add_argument('--input_file', default='/home/wdl/datasets/LM/text8')
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--embed_size', type=int, default=100)
  parser.add_argument('--skip_window', type=int, default=1)
  parser.add_argument('--num_sampled', type=int, default=64)
  parser.add_argument('--nepochs', type=int, default=1)
  parser.add_argument('--log_interval', type=int, default=1000)
  parser.add_argument('--learning_rate', type=float, default=1.0)

  args = parser.parse_args()

  dataset = Dataset(args.input_file)
  model = Word2vec(args.batch_size, dataset.vocab_size, args.embed_size)

  nce_weight = tf.Variable(tf.truncated_normal([dataset.vocab_size, args.embed_size],\
    stddev=1.0 / (args.embed_size ** 0.5)))
  nce_bias = tf.Variable(tf.zeros([dataset.vocab_size]))

  loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,\
    biases=nce_bias,\
    inputs=model.embed,\
    labels=model.target_words,\
    num_sampled=args.num_sampled,\
    num_classes=dataset.vocab_size))
  
  optimizer = tf.train.GradientDescentOptimizer(args.learning_rate).\
    minimize(loss, global_step=model.global_step)

  saver = tf.train.Saver()
  os.mkdir('checkpoints')

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  with tf.device('/gpu:1'), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))\
    as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
    if ckpt and ckpt.model_check_point_path:
      saver.restore(sess, chpt.model_check_point_path)

    total_loss = 0.0
    batch_index = 0
    for center_batch, target_batch in dataset.get_batch(args.batch_size,\
      args.skip_window, args.nepochs):
      batch_index += 1
      batch_loss, _ = sess.run([loss, optimizer],\
        feed_dict={model.center_words : center_batch,\
        model.target_words : target_batch})
      total_loss += batch_loss
      if batch_index % args.log_interval == 0:
        logging.info("Batch : {}, loss : {:5.1f}".format(batch_index,\
          total_loss/args.log_interval))
        total_loss = 0.0
        saver.save(sess, 'checkpoints/skip-gram', batch_index)