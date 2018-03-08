#!/usr/bin/python
# coding=utf-8
import numpy as np
from dataset import Dataset
from tensorflow.contrib import learn
import re

class SSTDataset(Dataset):
    def __init__(self, positive_data_file, negative_data_file,
        validation_perc=0.0, vocab_processor=None):
        '''
        Load MR polarity data from files, split the data into words and
        egnerates labels
        '''
        Dataset.__init__(self)
        # Load data from files
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = map(lambda x: x.strip(), positive_examples)
        negative_exmpales = list(open(negative_data_file, "r").readlines())
        negative_exmpales = map(lambda x: x.strip(), negative_exmpales)

        # Split by words
        all_data = positive_examples + negative_exmpales
        all_data = map(self.clean_str, all_data)
        all_data = np.array(all_data)

        max_document_length = max([len(x.split(" ")) for x in all_data])
        if vocab_processor is None:
            self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
            all_data = np.array(list(self.vocab_processor.fit_transform(all_data)))
        else:
            self.vocab_processor = vocab_processor
            all_data = self.vocab_processor.transform(all_data)

        # Labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_exmpales]

        all_labels = np.concatenate([positive_labels, negative_labels], 0)
        all_labels = np.array(all_labels)

        perm = np.arange(len(all_data))
        np.random.shuffle(perm)
        all_data = all_data[perm]
        all_labels = all_labels[perm]

        train_valid_split_cnt = int(len(all_data) * (1.0 - validation_perc))
        self.train_data = all_data[:train_valid_split_cnt]
        self.train_label = all_labels[:train_valid_split_cnt]
        self.val_data = all_data[train_valid_split_cnt:]
        self.val_label = all_labels[train_valid_split_cnt:]
        print("Max Document Length {len}".format(len=max_document_length))
        print("Vocabulary Size: {len}".format(len=len(self.vocab_processor.vocabulary_)))
        print("Train/Validation Split: {:d}/{:d}".format(len(self.train_data), len(self.val_data)))

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()



