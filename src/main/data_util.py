import os
import shutil
import re
from random import shuffle


class SimpleVocabularyEncoder(object):
    def __init__(self, vocab):
        self.__vocab = vocab
        self.__encoder_dict = {}
        self.__decoder_dict = {}
        i = 0
        for c in self.__vocab:
            self.__encoder_dict[c] = i
            self.__decoder_dict[i] = c
            i += 1

        assert len(self.__decoder_dict) == len(self.__encoder_dict) == len(self.__vocab)

    def encode(self, c):
        return self.__encoder_dict[c]

    def decode(self, i):
        return self.__decoder_dict[i]

    def encode_sequence(self, seq):
        encoded = []
        for c in seq:
            assert c in self.__encoder_dict
            encoded.append(self.__encoder_dict[c])
        return encoded

    def decode_sequence(self, seq):
        decoded = []
        for c in seq:
            assert c in self.__decoder_dict
            decoded.append(self.__decoder_dict[c])
        return decoded

    def get_num_classes(self):
        return len(self.__vocab)

    def add_vocab_symbol(self, symbol):
        self.__vocab.add(symbol)
        self.__encoder_dict[symbol] = len(self.__encoder_dict)
        self.__decoder_dict[len(self.__decoder_dict)] = symbol

        assert len(self.__decoder_dict) == len(self.__encoder_dict) == len(self.__vocab)


# TODO why is cwd different in debug and run mode?!
class DataUtil(object):
    path_to_cmudict_repo = '../../data/cmudict/'
    cmudict_file_name = 'cmudict.dict'
    path_to_dict = path_to_cmudict_repo + cmudict_file_name
    phoneme_symbols_file_name = 'cmudict.symbols'

    def __init__(self, path_to_cmudict_repo='../../data/cmudict/'):
        self.path_to_cmudict_repo = path_to_cmudict_repo
        return

    def get_cmudict(self, multiple_entries=False, shuffled=True, stress_indicators=True):
        """
        loads the cmudict data from disk
        :return: a dictionary from word to phonemes
        """
        assert os.path.isfile(self.path_to_dict), 'Cannot find cmudict file at: ' + self.path_to_dict

        cmudict_raw = open(self.path_to_dict).readlines()
        cmudict = {}
        for line in cmudict_raw:
            if not multiple_entries and '(' in line:
                continue
            # remove comments
            if '#' in line:
                line = line[:line.index('#')]
            line = line.split()
            gra = line[0]
            pho = line[1:]
            if not stress_indicators:
                regex = re.compile("\d")
                pho_no_stress = []
                for p in pho:
                    pho_no_stress.append(re.sub(regex, "", p))
                pho = pho_no_stress
            cmudict[gra] = pho

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

    def get_sequential_cmudict_for_lstm(self, stress_indicators=False):
        # load cmudict without multiple entries and shuffled
        cmudict = self.get_cmudict(multiple_entries=False, shuffled=True, stress_indicators=stress_indicators)

        # generate vocabularies
        # phonemes
        phoneme_vocab = set()
        for v in cmudict.values():
            for p in v:
                phoneme_vocab.add(p)
        # graphemes
        grapheme_vocab = set()
        for k in cmudict.keys():
            for g in k:
                grapheme_vocab.add(g)

        # each char will be a 'word' in graphemes and each phoneme in phonemes will be a 'word'
        grapheme_sequences = list()
        phoneme_sequnences = list()

        max_grapheme_seq_len = 0
        max_phoneme_seq_len = 0

        # generate the grapheme and phoneme sequences so that
        # grapheme_sequences[i] matches phoneme_sequences[i]
        for g, p in cmudict.items():
            current_grapheme = list()
            for c in g:
                current_grapheme.append(c)
            grapheme_sequences.append(current_grapheme)
            if len(current_grapheme) > max_grapheme_seq_len:
                max_grapheme_seq_len = len(current_grapheme)

            current_phoneme = list()
            for c in p:
                current_phoneme.append(c)
            phoneme_sequnences.append(current_phoneme)
            if len(current_phoneme) > max_phoneme_seq_len:
                max_phoneme_seq_len = len(current_phoneme)

        return grapheme_vocab, phoneme_vocab, grapheme_sequences, phoneme_sequnences, max_grapheme_seq_len, max_phoneme_seq_len

    def generate_cmudict_files_for_tf_nmt(self, path_to_store_files='../../data/cmudictNMT/', training_percent=0.98,
                                          test_percent=0.01,
                                          dev_percent=0.01):
        """
        prepares the cmudict for the Tensorflow NMT model (https://github.com/tensorflow/nmt)

        :param training_percent: percentage of training set size
        :param test_percent:  percentage of test set size
        :param dev_percent: percentage of dev set size
        :param path_to_store_files: path to the directory where the created files get saved
        :return file names of generated files
        """

        # parameters to use with the NMT model
        nmt_vocab_prefix = 'vocab'
        nmt_train_prefix = 'train'
        nmt_test_prefix = 'test'
        nmt_dev_prefix = 'dev'
        grapheme_suffix = 'gra'
        phonemes_suffix = 'pho'
        # file names
        nmt_grapheme_vocab_file_name = path_to_store_files + nmt_vocab_prefix + "." + grapheme_suffix
        nmt_phoneme_vocab_file_name = path_to_store_files + nmt_vocab_prefix + "." + phonemes_suffix
        nmt_grapheme_training_file_name = path_to_store_files + nmt_train_prefix + "." + grapheme_suffix
        nmt_phoneme_training_file_name = path_to_store_files + nmt_train_prefix + "." + phonemes_suffix
        nmt_grapheme_test_file_name = path_to_store_files + nmt_test_prefix + "." + grapheme_suffix
        nmt_phoneme_test_file_name = path_to_store_files + nmt_test_prefix + "." + phonemes_suffix
        nmt_grapheme_dev_file_name = path_to_store_files + nmt_dev_prefix + "." + grapheme_suffix
        nmt_phoneme_dev_file_name = path_to_store_files + nmt_dev_prefix + "." + phonemes_suffix

        if not os.path.isdir(path_to_store_files):
            os.mkdir(path_to_store_files)

        grapheme_vocab, phoneme_vocab, grapheme_sequences, phoneme_sequences, _, _ = self.get_sequential_cmudict_for_lstm(
            stress_indicators=False)

        # create the vocabulary files
        #  for phonemes vocab already there, so just do a simple copy
        shutil.copy(self.path_to_cmudict_repo + self.phoneme_symbols_file_name, nmt_phoneme_vocab_file_name)
        #  for graphemes get all the unique chars
        grapheme_vocab_file = open(nmt_grapheme_vocab_file_name, 'w')
        for uc in grapheme_vocab:
            print(uc, file=grapheme_vocab_file)
        grapheme_vocab_file.close()

        # calc train, test and dev set sizes
        assert len(grapheme_sequences) == len(phoneme_sequences)
        train_size = int(len(grapheme_sequences) * training_percent)
        test_size = int(len(grapheme_sequences) * test_percent)
        dev_size = int(len(grapheme_sequences) * dev_percent)

        # create the training files
        grapheme_training_file = open(nmt_grapheme_training_file_name, 'w')
        phoneme_training_file = open(nmt_phoneme_training_file_name, 'w')
        for w in grapheme_sequences[:train_size]:
            for c in w:
                print(c, end=' ', file=grapheme_training_file)
            print('', file=grapheme_training_file)
        for w in phoneme_sequences[:train_size]:
            for c in w:
                print(c, end=' ', file=phoneme_training_file)
            print('', file=phoneme_training_file)
        grapheme_training_file.close()
        phoneme_training_file.close()

        # create the test files
        grapheme_test_file = open(nmt_grapheme_test_file_name, 'w')
        phoneme_test_file = open(nmt_phoneme_test_file_name, 'w')
        for w in grapheme_sequences[train_size:train_size + test_size]:
            for c in w:
                print(c, end=' ', file=grapheme_test_file)
            print('', file=grapheme_test_file)
        for w in phoneme_sequences[train_size:train_size + test_size]:
            for c in w:
                print(c, end=' ', file=phoneme_test_file)
            print('', file=phoneme_test_file)
        phoneme_test_file.close()
        grapheme_test_file.close()

        # create the dev files
        grapheme_dev_file = open(nmt_grapheme_dev_file_name, 'w')
        phoneme_dev_file = open(nmt_phoneme_dev_file_name, 'w')
        for w in grapheme_sequences[train_size + test_size:]:
            for c in w:
                print(c, end=' ', file=grapheme_dev_file)
            print('', file=grapheme_dev_file)
        for w in phoneme_sequences[train_size + test_size:]:
            for c in w:
                print(c, end=' ', file=phoneme_dev_file)
            print('', file=phoneme_dev_file)
        grapheme_dev_file.close()
        phoneme_dev_file.close()

        return nmt_grapheme_training_file_name, nmt_phoneme_training_file_name, \
               nmt_grapheme_dev_file_name, nmt_phoneme_dev_file_name, \
               nmt_grapheme_test_file_name, nmt_phoneme_test_file_name, \
               nmt_grapheme_dev_file_name, nmt_phoneme_dev_file_name


if __name__ == '__main__':
    print('current working dir:' + os.getcwd())

    d_util = DataUtil()
    # cmudict = d_util.load_data()
    d_util.generate_cmudict_files_for_tf_nmt()
    # grapheme_vocab, phoneme_vocab, grapheme_word_list, phoneme_word_list, max_grapheme_seq_len, max_phoneme_seq_len = d_util.get_sequential_cmudict_for_lstm()
