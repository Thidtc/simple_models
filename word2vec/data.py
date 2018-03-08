# coding=utf-8
import random
import numpy as np
import logging

class Dictionary(object):
  def __init__(self):
    self.word2id = {}
    self.id2word = []
  def add_word(self, word):
    if not word in self.word2id:
      self.word2id[word] = len(self.id2word)
      self.id2word.append(word)
    return self.word2id[word]

  def __len__(self):
    return len(self.id2word)

class Dataset(object):
  def __init__(self, filename):
    self.dic = Dictionary()
    idxes = []
    logging.info("Start load {}".format(filename))
    with open(filename, 'r') as f:
      words = f.readline().split()
      for word in words:
        idxes.append(self.dic.add_word(word))
      del words 
    self.words = idxes
    logging.info("Data size: {}".format(len(self.words)))
    logging.info("Vocabulary size: {}".format(len(self.dic)))
  
  @property
  def vocab_size(self):
    return len(self.dic)

  def get_batch(self, batch_size, skip_window, nepochs, shuffle=True):
    '''
    generate batch data
    '''
    iter = self.generate_sample(self.words, skip_window, nepochs, shuffle)
    try:
      while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for i in range(batch_size):
          center_batch[i], target_batch[i] = next(iter)
        yield center_batch, target_batch
    except Exception as e:
      pass

  def generate_sample(self, words, skip_window, nepochs, shuffle):
    '''
    Genrate one sample
    '''
    for i in range(nepochs):
      if shuffle:
        random.shuffle(words)
      for idx, center_word in enumerate(words):
        context_size = random.randint(1, skip_window)
        for target_word in words[max(0, idx - context_size):idx]:
          yield center_word, target_word
        for target_word in words[idx + 1 : min(len(words) - 1,\
          idx + context_size + 1)]:
          yield center_word, target_word

if __name__ == '__main__':
  dataset = Dataset('/home/wdl/datasets/LM/text8')
  iter = dataset.generate_sample(dataset.words, 3, 1, False)
  next(iter)