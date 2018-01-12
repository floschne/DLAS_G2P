import os
import shutil
from random import shuffle


# TODO why is cwd different in debug and run mode?!

class data_util(object):
    path_to_cmudict_repo = '../../data/cmudict/'
    cmudict_file_name = 'cmudict.dict'
    path_to_dict = path_to_cmudict_repo + cmudict_file_name
    phoneme_symbols_file_name = 'cmudict.symbols'

    def __init__(self, path_to_cmudict_repo='../../data/cmudict/'):
        self.path_to_cmudict_repo = path_to_cmudict_repo
        return

    def load_data(self, multipleEntries=True, shuffled=True):
        """
        loads the cmudict data from disk
        :return: a dictionary from word to phonemes
        """
        assert os.path.isfile(self.path_to_dict), 'Cannot find cmudict file at: ' + self.path_to_dict

        cmudict_raw = open(self.path_to_dict).readlines()
        cmudict = {}
        for line in cmudict_raw:
            if not multipleEntries and '(' in line:
                continue
            line = line.split()
            cmudict[line[0]] = line[1:]

        # shuffle the dict
        cmudict_shuffled = dict()
        if shuffled:
            keys = list(cmudict.keys())
            shuffle(keys)
            for k in keys:
                cmudict_shuffled[k] = cmudict[k]

        if not shuffled:
            return cmudict
        else:
            return cmudict_shuffled

    def prepare_cmudict_for_tf_nmt(self, path_to_store_files='../../data/cmudictNMT/', training_percent=0.98,
                                   test_percent=0.01,
                                   dev_percent=0.01):
        """
        prepares the cmudict for the Tensorflow NMT model (https://github.com/tensorflow/nmt)

        :param training_percent: percentage of training set size
        :param test_percent:  percentage of test set size
        :param dev_percent: percentage of dev set size
        :param path_to_store_files: path to the directory where the created files get saved
        :return: prefixes of the different sets and suffixes of the grapheme or phoneme (i.e. the file names of the sets)
        """

        # parameters to use with the NMT model
        train_prefix = path_to_store_files + 'train'
        vocab_prefix = path_to_store_files + 'vocab'
        test_prefix = path_to_store_files + 'test'
        dev_prefix = path_to_store_files + 'dev'
        grapheme_suffix = 'gra'
        phonemes_suffix = 'pho'

        if not os.path.isdir(path_to_store_files):
            os.mkdir(path_to_store_files)

        # load cmudict without multiple entries
        cmudict = self.load_data(multipleEntries=False, shuffled=True)

        # create the vocabulary files
        #  for phonemes vocab already there, so just do a simple copy
        phonemes_vocab_file = vocab_prefix + "." + phonemes_suffix
        shutil.copy(self.path_to_cmudict_repo + self.phoneme_symbols_file_name, phonemes_vocab_file)
        #  for graphemes get all unique chars
        grapheme_vocab_file = open(vocab_prefix + "." + grapheme_suffix, 'w')
        unique_chars = set()
        for k in cmudict.keys():
            for c in k:
                unique_chars.add(c)
        for uc in unique_chars:
            print(uc, file=grapheme_vocab_file)

        grapheme_vocab_file.close()

        # each char will be a word in graphemes and each phoneme in phonemes will be a word
        grapheme_word_list = list()
        phoneme_word_list = list()
        for g, p in cmudict.items():
            current_grapheme = list()
            for c in g:
                current_grapheme.append(c)
            grapheme_word_list.append(current_grapheme)

            current_phoneme = list()
            for c in p:
                current_phoneme.append(c)
            phoneme_word_list.append(current_phoneme)

        # calc train, test and dev set sizes
        assert len(grapheme_word_list) == len(phoneme_word_list)
        train_size = int(len(grapheme_word_list) * training_percent)
        test_size = int(len(grapheme_word_list) * test_percent)
        dev_size = int(len(grapheme_word_list) * dev_percent)

        # create the training files
        grapheme_training_file = open(train_prefix + "." + grapheme_suffix, 'w')
        phoneme_training_file = open(train_prefix + "." + phonemes_suffix, 'w')
        for w in grapheme_word_list[:train_size]:
            for c in w:
                print(c, end=' ', file=grapheme_training_file)
            print('', file=grapheme_training_file)
        for w in phoneme_word_list[:train_size]:
            for c in w:
                print(c, end=' ', file=phoneme_training_file)
            print('', file=phoneme_training_file)

        grapheme_training_file.close()
        phoneme_training_file.close()

        # create the test files
        grapheme_test_file = open(test_prefix + "." + grapheme_suffix, 'w')
        phoneme_test_file = open(test_prefix + "." + phonemes_suffix, 'w')
        for w in grapheme_word_list[train_size:train_size + test_size]:
            for c in w:
                print(c, end=' ', file=grapheme_test_file)
            print('', file=grapheme_test_file)
        for w in phoneme_word_list[train_size:train_size + test_size]:
            for c in w:
                print(c, end=' ', file=phoneme_test_file)
            print('', file=phoneme_test_file)
        phoneme_test_file.close()
        grapheme_test_file.close()

        # create the dev files
        grapheme_dev_file = open(dev_prefix + "." + grapheme_suffix, 'w')
        phoneme_dev_file = open(dev_prefix + "." + phonemes_suffix, 'w')
        for w in grapheme_word_list[train_size + test_size:]:
            for c in w:
                print(c, end=' ', file=grapheme_dev_file)
            print('', file=grapheme_dev_file)
        for w in phoneme_word_list[train_size + test_size:]:
            for c in w:
                print(c, end=' ', file=phoneme_dev_file)
            print('', file=phoneme_dev_file)
        grapheme_dev_file.close()
        phoneme_dev_file.close()

        return vocab_prefix, train_prefix, dev_prefix, test_prefix, grapheme_suffix, phonemes_suffix


def test():
    print('current working dir:' + os.getcwd())

    d_util = data_util()
    # cmudict = d_util.load_data()
    d_util.prepare_cmudict_for_tf_nmt()
    return


if __name__ == '__main__':
    test()
