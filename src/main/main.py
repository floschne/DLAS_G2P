import os
import tensorflow as tf


def prepare_data(path_to_dict='../../data/cmudict/cmudict.dict'):
    assert os.path.isfile(path_to_dict), 'Cannot find cmudict file at: ' + path_to_dict

    cmudict_raw = open(path_to_dict).readlines()
    cmudict = {}
    for line in cmudict_raw:
        line = line.split()
        cmudict[line[0]] = line[1:]

    assert len(cmudict) == len(cmudict_raw), 'Error while reading dictionary!'

    return cmudict


def build_model():

    return

cmudict = prepare_data()
print(cmudict['tree'])
