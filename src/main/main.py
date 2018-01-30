import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np
import importlib as imp

du = imp.import_module("data_util")
du = du.data_util()

cmudict = du.get_cmudict()

print(cmudict["tree"])

print('Hello G2P')
