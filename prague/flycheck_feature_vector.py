'''
Contains functions for creating and manipulating NumPy ndarrays that represent
(partial) binary feature vectors
'''

import numpy as np
import scipy.special
import random

from utility import composable_put, composable_put_along_axis

INT8 = np.int8

#################
# Pseudo-typing #
#################

def wf_pfv(v):
    '''
    Indicates whether v is a well-formed partially-specified feature vector.
    '''
    allowedValues = {-1,0,1}
    return all([x in allowedValues for x in v])


def wf_tfv(v):
    '''
    Indicates whether v is a well-formed totally-specified feature vector.
    '''
    allowedValues = {-1,1}
    return all([x in allowedValues for x in v])


# def uniquify(ndarray_iterable):
#     tuples = [tuple(a) for a in ndarray_iterable]
#     s = set(tuples)
#     arrays = [np.array(t) for t in s]
#     return arrays


################################################################
# Construction of partial feature vectors and sets and objects #
################################################################


def make_random_pfv(num_features):
    '''Returns a random partial feature vector with num_features features.

    Return type is an ndarray.
    '''
    return np.random.randint(3, size=num_features, dtype=INT8) - 1


def make_random_fspfv(num_features):
    '''Returns a random fully specified feature vector with num_features
    features.

    Return type is an ndarray.
    '''
    return np.random.randint(2, size=num_features, dtype=INT8)


def zero_to_minus_one(u):
    '''Accepts a 1D ndarray and returns a copy with every 0 mapped to -1.'''
    return np.array([x if x == 1 else -1 for x in u], dtype=INT8)


def make_random_objects(num_objects, num_features, as_ndarray=True,
                        unique=False):
    '''By default, makes num_objects random objects, all with
    num_features features, and all fully specified; not guaranteed to be unique.
    If unique is True, then returns *at most* num_objects random objects.

    If as_ndarray is True, every row is an object vector.
    If as_ndarray is False, the set of objects is returned as a tuple.

    The set of objects is sorted in lexicographic order.
    '''
    num_possible_objects = 2 ** num_features
    if num_objects > num_possible_objects:
        raise Exception((f"Number of desired objects ({num_objects})"
                         f" must be <= 2^[number of features]"
                         f" ({num_possible_objects})."))

    objects = [tuple(make_random_fspfv(num_features))
               for each in range(num_objects)]

    objects = sorted(objects)
    if unique:
        objects = set(objects)
    objects = tuple(map(lambda o: zero_to_minus_one(o),
                        objects))

    if not as_ndarray:
        return objects
    return np.array(objects, dtype=INT8)


def load_object_vectors(filepath):
    '''
    Loads an ndarray from an .npy file; each row is assumed to be an object.
    '''
    return np.load(filepath)
