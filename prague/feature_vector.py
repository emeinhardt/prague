'''
Contains functions for creating and manipulating NumPy ndarrays that represent
(partial) binary feature vectors = (balanced) ternary feature vectors.
'''

import numpy as np
import scipy.special
import scipy.spatial.distance
import random

from funcy import cat, lmap
from functools import reduce

import itertools

from hashlib import sha1

import collections

INT8 = np.int8


################
# VECTOR ABUSE #
################


#from https://stackoverflow.com/a/11146645
def cartesian_product(*arrays):
    '''
    Given n one dimensional vectors, construct their cartesian product analogue.
    '''
    la    = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def cartesian_product_stack(stack_a, stack_b):
    '''
    Given two stacks of vectors (n x a and m x b), returns two stacks of
    vectors (n x m x a, n x m x b) where the pair of vectors at row i of each
    stack represents an element of the cartesian product of the input stacks of
    vectors.
    '''
    n, a = stack_a.shape
    m, b = stack_b.shape
    # n = stack_a.shape[0]
    # m = stack_b.shape[0]
    left  = np.repeat(stack_a, m, axis=0)
    right = np.tile(stack_b, (n, 1))
    assert left.shape[0]  == n*m, f"left shape 0th dimension should be n*m={n*m}, but instead is {left.shape[0]}"
    assert right.shape[0] == n*m, f"right shape 0th dimension should be n*m={n*m}, but instead is {right.shape[0]}"
    # return left, right
    left  = np.reshape(left,  (n,m,a))
    right = np.reshape(right, (n,m,b))
    return left, right


def prefixes(arr):
    '''
    Given an array (or stack), returns the 'prefixes' along the first dimension:
    Ex: Given
        x = np.array([0,1,2])
    then
        prefixes(x)
        => (array([]), array([0]), array([0,1]), array([0,1,2]))
    '''
    return tuple([arr[:-(arr.shape[0]-i)] for i in np.arange(arr.shape[0])] + [arr])


#################################
# HASHING BALANCED TERNARY PFVS #
#################################


UNSIGNED_PRECISIONS = np.array([8, 16, 32, 64]).astype(np.uint64)
UNSIGNED_MAX_INTS = (2 ** UNSIGNED_PRECISIONS) - 1
MAX_M_FOR_HASHING = 40  # 3^40 < 2^64, but 3^41 > 2^64....

# 64 * np.log(2) / np.log(39) is slightly more than 40
# 23 * np.log(3) / np.log(20) is slightly more than 36
#                                         => we'll need np.uint64 precision...


def ternary_pfv_to_trits(pfv):
    '''
    Converts a balanced ternary pfv to base-3 ('unbalanced trinary' or
    'unbalanced ternary') element-wise:
      [-1, 1, 0] => [0, 2, 1]
    '''
    return pfv + 1


def trits_to_ternary_pfv(trits):
    '''
    Converts a base-3 ndarry to balanced ternary, element-wise:
      [0, 2, 1] => [-1, 1, 0]
    '''
    return trits - 1


#slightly adapted from https://stackoverflow.com/a/20228144
def int2base(x, base, size=None, order='decreasing', dtype=INT8):
    '''
    Converts an int (or stack of ints) x in base-10 to an array
    (or stack of arrays) representing the equivalent in the argument 
    base.
    
    E.g.
      int2base(43244977933, 10) 
        => array([4, 3, 2, 4, 4, 9, 7, 7, 9, 3, 3])
      int2base(43244977933, 3) 
        => array([1 1 0 1 0 1 2 1 2 1 1 0 0 1 1 1 1 1 0 1 1 0 1])
      int2base(np.array([53754432896 48908987687]), 3)
        => array([[1 2 0 1 0 2 0 2 0 2 0 1 1 0 0 1 0 1 0 0 0 0 2]
                  [1 1 2 0 0 0 2 0 1 1 2 2 2 0 0 0 0 0 0 2 1 0 2]])
    '''
    x = np.asarray(x).astype(np.uint64)
    if size is None:
        size = int(np.ceil(np.log(np.max(x))/np.log(base)))
    if order == "decreasing":
        powers = base ** np.arange(size - 1, -1, -1)
    else:
        powers = base ** np.arange(size)
    digits = (x.reshape(x.shape + (1,)) // powers) % base
    return digits.astype(dtype)


def trits_to_int(trits):
    '''
    Converts a vector representing a base-3 (unbalanced trinary) number (or 
    stack of such vectors) to the corresponding (stack of) base-10 integer(s).
    
    E.g.
      [0 1 2 0] corresponds to 1 x (3²) + 2 x (3¹) + 0 x (3⁰) = 9 + 6 = 15
    so 
      trits_to_int(np.array([0, 1, 2, 0])) => 15
    And 
      trits_to_int(np.array([[0, 1, 2, 0], [0, 0, 1, 1]])) => np.array([15, 4])
    '''
    m = trits.shape[-1]
    exponents  = np.flip(np.arange(m)).astype(np.uint64)
    my_base    = 3
    powers     = np.power(np.array([my_base], dtype=np.uint64),
                         exponents)
#     base10_int = np.dot(trits, powers).astype(np.uint64)
    base10_int = np.matmul(trits.astype(np.uint64), powers, dtype=np.uint64)
    return base10_int


def int_to_trits(base10_int, m=None):
    '''
    Converts a base-10 integer (or stack of them) to their equivalent base-3 
    (unbalanced trinary) representation (or stack of them).
    
    E.g. 
     int_to_trits(43244977933) 
       => np.array([1 1 0 1 0 1 2 1 2 1 1 0 0 1 1 1 1 1 0 1 1 0 1])
     int_to_trits(np.array([53754432896, 48908987687])
       => [[1 2 0 1 0 2 0 2 0 2 0 1 1 0 0 1 0 1 0 0 0 0 2]
           [1 1 2 0 0 0 2 0 1 1 2 2 2 0 0 0 0 0 0 2 1 0 2]]
     int_to_trits(16) => np.array([1,  2,  1])
     int_to_trits(16, m=4) => np.array([0,  1,  2,  1])
    '''
    return int2base(base10_int, base=3, size=m)            


def hash_ternary_pfv(pfv):
    '''
    Converts a balanced ternary pfv to a unique unsigned int corresponding
    to a base-10 representation.
    '''
    return trits_to_int(ternary_pfv_to_trits(pfv))


def decode_hash(hash_int, m=None):
    '''
    Converts an unsigned 64-bit integer representing the hash of a balanced
    ternary array back into that array.
    '''
    return trits_to_ternary_pfv(int_to_trits(hash_int, m=m))


###########
# UTILITY #
###########


def composable_put(a, ind, v, mode='raise', copy_arg=True):
    '''A version of np.put that supports functional programming (i.e.
    compositionality, referential transparency) by returning the array it
    modifies and (by default) not mutating its primary argument.

    See the documentation for the original function for more details.
    '''
    if copy_arg:
        my_a = a.copy()
    else:
        my_a = a
    np.put(a=my_a, ind=ind, v=v, mode=mode)
    return my_a


def composable_put_along_axis(arr, indices, values, axis=None, copy_arg=True):
    '''A version of np.put_along_axis that supports functional programming
    (i.e. compositionality, referential transparency) by returning the array it
    modifies and (by default) not mutating its primary argument.

    See the documentation for the original function for more details.
    '''
    if copy_arg:
        my_arr = arr.copy()
    else:
        my_arr = arr
    np.put_along_axis(my_arr, indices, values, axis=axis)
    return my_arr


#tweaked version of http://machineawakening.blogspot.com/2011/03/making-numpy-ndarrays-hashable.html
class HashableArray(object):
    r'''Hashable wrapper for ndarray objects.

        Instances of ndarray are not hashable, meaning they cannot be added to
        sets, nor used as keys in dictionaries. This is by design - ndarray
        objects are mutable, and therefore cannot reliably implement the
        __hash__() method.

        The hashable class allows a way around this limitation. It implements
        the required methods for hashable objects in terms of an encapsulated
        ndarray object. This can be either a copied instance (which is safer)
        or the original object (which requires the user to be careful enough
        not to modify it).
    '''

   
    def __init__(self, arr, tight=False):
        r'''Creates a new hashable object encapsulating an ndarray.

            arr
                The ndarray to wrap.

            tight
                Optional. If True, a copy of the input ndaray is created.
                Defaults to False.
        '''
        self.__tight   = tight
        self.__wrapped = np.array(arr) if tight else arr
        self.__hash    = int(sha1(arr.view(np.uint8)).hexdigest(), 16)


    def __eq__(self, other):
        return np.all(self.__wrapped == other.__wrapped)


    def __hash__(self):
        return self.__hash


    def unwrap(self):
        r'''Returns the encapsulated ndarray.

            If the wrapper is "tight", a copy of the encapsulated ndarray is
            returned. Otherwise, the encapsulated ndarray itself is returned.
        '''
        if self.__tight:
            return np.array(self.__wrapped)

        return self.__wrapped
    
    
    def __str__(self):
        return str(self.unwrap())
    
    
    def __repr__(self):
        return f"Hashable({self.unwrap().__repr__()})"


def from_feature_dict(d, feature_seq):
    '''
    Given a feature dictionary and an ordering on features, returns a ternary
    vector version of the dictionary.
    '''
    value_map     = {'0':0, '-':-1, '+':1}
    value_mapping = lambda v: value_map[v]
    return np.array([value_mapping(d[f]) if f in d else 0 for f in feature_seq], dtype=INT8)


def to_feature_dict(feature_seq, u, value_map=None):
    '''
    Given a sequence of features and a pfv u, creates an equivalent feature 
    dictionary.
    Assumes feature_seq is ordered appropriately relative to u.

    If value_map is None, the default (0->'0', 1->'+', -1->'-') will be used.
    '''
    n_features = len(feature_seq)
    n_vals     = u.shape[0]
    assert n_features == n_vals, f"Num features does not match length of u: {n_features} vs. {n_vals}"

    if value_map is None:
        value_map = {-1:'-', 1:'+', 0:'0'}

    mapped_vals = [value_map[val]
                   for val in u]

    return dict(zip(feature_seq, mapped_vals))


def to_spe(feature_seq=None, v=None, d=None):
    '''
    Given a a feature dictionary d or both a pfv v and a (complete) sequence of
    features, this returns a string with equivalent traditional SPE-style
    notation.

    If a feature dictionary is provided AND a feature sequence is provided,
    then the feature sequence will determine the ordering of features in the
    resulting string. Note that in this case, the feature sequence need only
    mention the specified features of d.
    '''
    if d is not None:
        specified_features = {k for k in d if d[k] != '0'}
        if feature_seq is not None:
            missing_from_feature_seq = {f for f in specified_features if f not in feature_seq}
            if len(missing_from_feature_seq) > 0:
                raise Exception(f'Specified features of d are missing from feature_seq:\n\t{missing_from_feature_seq}')
        else:
            feature_seq = sorted(list(specified_features))

        # value_map = {-1:'-', 1:'+', 0:'0'}
        # to_val = lambda v:value_map[v]

        # s = ' '.join([f"{to_val(d[f])}{f}" for f in feature_seq if d[f] != '0'])
        s = "[" + ' '.join([f"{d[f]}{f}" for f in feature_seq if d[f] != '0']) + "]"
        return s
    else:
        d = to_feature_dict(feature_seq, v)
        return to_spe(d=d)


def from_spe(s, features=None):
    '''
    Given a one-line string with an SPE-style feature vector, returns an
    equivalent feature dictionary. In the event you want unspecified features
    to be explicitly present and unspecified (in general, you should), you must
    provide a set of features.
    '''
    assert s[0] == '['
    assert s[-1] == ']'
    no_brackets  = s[1:-1]
    split        = [each for each in no_brackets.split(' ') if each != '']
    space_joined = []
    for each in split:
        if each[0] in {'+','-','0'}:
            space_joined.append(each)
        else:
            space_joined[-1] = space_joined[-1] + ' ' + each
    parse_term = lambda t: (t[1:], t[0])
    no_terms = len(space_joined) == 0

    if no_terms:
        return {f:'0' for f in features}
    spec_map = {f:v for f,v in map(parse_term, space_joined)}
    assert all(v in {'0','-','+'} for v in spec_map.values()), f"Illegal feature value in parsed vector:\n{spec_map}"
    if features is not None:
        d = dict()
        for f in features:
            if f not in spec_map:
                d[f] = '0'
            else:
                d[f] = spec_map[f]
        return d
    else:
        return spec_map


def stack_to_set(M):
    '''
    Given a k x m ndarray representing a stack of k pfvs, this returns the
    corresponding set of HashableArrays.
    '''
    return set([HashableArray(v) for v in M])


def hashableArrays_to_stack(arrs):
    '''
    Given a k-length set of m-length HashableArrays representing a stack of k 
    pfvs, this returns a corresponding stack of ndarrays.
    '''
    return np.array([v.unwrap() for v in arrs], dtype=INT8)


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

def make_zero_pfv(num_features):
    '''Returns the vector of length m = num_features of only zeros.
    
    Return type is an ndarray.
    '''
    return np.zeros((num_features,),dtype=INT8)


def make_ones_pfv(num_features):
    '''Returns the vector of length m = num_features of only ones.

    Return type is an ndarray.
    '''
    return np.ones((num_features,),dtype=INT8)


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


######################################
# Converting between representations #
######################################


def symbol_to_feature_vector(s, objectsAsDicts, objectsAsPFVs):
    '''
    Given 
     - a canonical sequence of objects as dicts
     - a canonical sequence of objects as pfvs with a matched ordering
    
    this function maps a symbol s to a pfv.
    '''
    mapping = {o['symbol']:objectsAsPFVs[i]
               for i,o in enumerate(objectsAsDicts)}
    return mapping[s]


def feature_vector_to_symbols(fv, objectsAsDicts, objectsAsPFVs):
    '''
    Given 
     - a canonical sequence of objects as dicts
     - a canonical sequence of objects as pfvs with a matched ordering
    
    this function maps a pfv to the set of symbols that match it.
    '''
    unique_objects_np = np.unique(objectsAsPFVs, axis=0)
    symbol_to_fv = {o['symbol']:objectsAsPFVs[i]
                    for i,o in enumerate(objectsAsDicts)}
    fv_to_symbols_map = {HashableArray(fv):{s for s in symbol_to_fv
                                            if np.array_equal(fv, 
                                                              symbol_to_fv[s])}
                         for fv in unique_objects_np}
    return fv_to_symbols_map[HashableArray(fv)]


def extension_to_symbols(ext, objectsAsDicts, objectsAsPFVs):
    '''
    Given 
     - an extension vector for some dict or pfv
     - an aligned sequence of objects as dictionaries
     - an aligned stack of pfvs
    
    this returns the set of symbols that have extension=ext.
    '''
    unique_objects_np = np.unique(objectsAsPFVs, axis=0)
    index_to_symbols = [feature_vector_to_symbols(fv, 
                                                  objectsAsDicts, 
                                                  objectsAsPFVs) 
                        for fv in unique_objects_np]
    return np.array(index_to_symbols)[extension.nonzero()[0]]
    

def symbol_to_feature_dict(symbol, objectsAsDicts):
    '''
    Given 
     - a symbol 
     - a sequence of objects as dictionaries
    
    this returns a sequence of dicts that match symbol.
    '''
    return [fd for fd in objectsAsDicts if fd['symbol'] == symbol]
    
    
def pfv_to_fd(pfv, feature_list):
    '''
    Given
     - a pfv
     - an aligned sequence listing features
    
    this returns a feature dictionary for the pfv.
    '''
    return to_feature_dict(feature_list, pfv)


######################################
# PFV DIFFERENCE AND SPE-STYLE RULES #
######################################


def hamming(u,v):
    '''
    Computes the (unnormalized) Hamming distance between 1D arrays u and v -
    i.e. the number of indices at which u and v differ.

    Basically a wrapper for scipy.spatial.distance.hamming. See that docstring
    for more details.
    '''
    assert u.shape[0] == v.shape[0], f"Shape mismatch: {u.shape} vs. {v.shape}"
    n = u.shape[0]
    return int(n * scipy.spatial.distance.hamming(u,v))


def delta_right(u,v):
    '''
    Calculates the ternary vectors m, b s.t.
        (m * u) + b = v
    where * denotes elementwise product.

    Note that
     - m describes where to flip values of u (viz. where m is negative) or
       unspecify u (viz. where m is 0),
     - b describes where to specify unspecified values of u (viz. where b is
       non-zero) and how they should be specified.
    '''
    #m can map specified values to specified values and specified values to
    # unspecified values
    #b is responsible for mapping unspecified values to specified ones
    # specified_mask = u != 0
    unspecified_mask = u == 0
    b = unspecified_mask * v
    m = u * (v - b) #will be 1 where u and (v-b) are same, -1 where different
    assert np.array_equal((m*u) + b , v)
    return m, b


def delta_down(u,v):
    '''
    Given v ∈ ↓u, calculates the indices of u that must be unspecified to yield
    v.
    '''
    m, b = delta_right(u,v)
    assert np.sum(b) == 0, f"v={v} must be in the lower closure of u={u}"
    indices_to_flip_mask = m == -1
    assert not np.any(indices_to_flip_mask), f"v={v} must be in the lower closure of u={u}"
    indices_to_unspecify_mask = m == 0
    indices_to_unspecify = indices_to_unspecify_mask.nonzero()[0]
    return indices_to_unspecify


def linear_transform(u, m, b):
    '''Returns (m*u) + b.'''
    return (m*u) + b


def despec(u, indices):
    '''Returns a copy of u with the given indices unspecified.'''
    return composable_put(u, indices, 0)


def priority_union(a,b):
    '''
    Treating pfvs as partial functions from feature labels or indices to the 
    Booleans, this computes the right priority union of a and b:
      a + b = c
    where
      c_i = b_i  if b_i ≠ 0
            a_i  otherwise
    '''
    return spe_update(a,b)


def right_inv_priority_union_old(c,b):
    '''
    Let + denote right priority union, where for some unknown pfv a
      a + b = c
    
    If / denotes 'right_inv_priority_union' = the right inverse of right 
    priority union, then 
      c / b = { a | a + b = c }
    where a,b,c are all ternary pfvs.
    
    At the pointwise/ternary value level,
      x / 0 = {x}
      x / x = ↑0
    In other words, the only case where '(c,b)' can be informative about 'a'
    is when 'b' is 0, in which case a=c.
    '''
    #TODO clean this up, replace for loop + insertion with matrix construction 
    # and multiplication
    assert c.shape[-1] == b.shape[-1], "the last dimension of b and c (the number of features) must be the same."
    m = c.shape[-1]
    allValues = np.array([+1,0,-1], dtype=np.int8)
#     print(f"c={c}")
#     print(f"b={b}")
    
    whereBIsNonZero = (b != 0)
    bNonZero_count  = l = np.count_nonzero(b)
    whereBIsZero    = (b == 0)
    if np.all(whereBIsZero):
        result = c.copy()
        if len(result.shape) == 1:
            result = np.expand_dims(result, 0)
        return result
    bZero_count     = k = m - bNonZero_count
#     print(f"l={l}\nk={k}")
    
    stableValuesFromC = c*whereBIsZero
#     print(f"stableValuesFromC={stableValuesFromC}")
    
    #if there are k <= m    indices where b is     0, then
    #   there are l = m - k indices where b is NOT 0
    # and we will need 3^k pfvs
#     result_shape = (3**bNonZero_count, m)
    result = np.tile(stableValuesFromC, [3**l,1])
#     print(f"result=\n{result}")

    filler_columns = [allValues for each in range(bNonZero_count)]
    filler_matrix  = cartesian_product(*filler_columns) # shape = (3^l, k)
#     print(f"filler_matrix=\n{filler_matrix}")
    
    index_to_insert_column    = whereBIsNonZero.nonzero()[0]
    index_of_column_to_insert = np.arange(filler_matrix.shape[-1])
    zipped                    = np.vstack([index_to_insert_column, 
                                           index_of_column_to_insert])
#     print(f"zipped=\n{zipped}")
    
    for each in zipped.T:
        index_for_insertion = each[0]
        index_to_insert     = each[1]
        result[:, index_for_insertion] = filler_matrix[:,index_to_insert]
#     print(f"result=\n{result}")
    
#     for a_prime in result:
#         actual_c = prague.priority_union(a_prime, b)
#         assert np.array_equal(actual_c, c), f"{a_prime} + {b} = {actual_c} ≠ {c}"
    
    if len(result.shape) == 1:
        result = np.expand_dims(result, 0)

    return result


def left_inv_priority_union_old(a,c):
    '''
    Let + denote right priority union, where for some unknown pfv b
      a + b = c
    
    If \ denotes 'left_inv_priority_union' = the left inverse of right 
    priority union, then 
      a \ c = { b | a + b = c }
    where a,b,c are all ternary pfvs.
    
    At the pointwise/ternary value level,
      x \ y, y < x = ⊥
      0 \ x        = {x}
      x \ x        = ↓x
      x \ y, y ≠ x = {y}
    In other words, 
      a_i = 0             -> b_i = c_i
      a_i ≠ 0 ∧ a_i ≠ c_i -> b_i = c_i
      a_i ≠ 0 ∧ a_i = c_i -> b_i = c_i ∨ 0
    '''
    #TODO clean this up, replace for loop + insertion with matrix construction 
    # and multiplication
    assert a.shape[-1] == c.shape[-1], "the last dimension of a and c (the number of features) must be the same."
    m         = c.shape[-1]
    plusZero  = np.array([+1,0], dtype=np.int8)
    minusZero = np.array([-1,0], dtype=np.int8)
#     print(f"a={a}")
#     print(f"c={c}")
    
    whereAIsZero    = a == 0
#     print(f"whereAIsZero=\n{whereAIsZero}")
    if np.all(whereAIsZero):
        result = c.copy()
        if len(result.shape) == 1:
            result = np.expand_dims(result, 0)
        return result
    whereAIsNonZero = a != 0
    whereANeqC      = a != c
#     print(f"whereAIsNonZero && whereANeqC=\n{(whereAIsNonZero & whereANeqC)}")
    whereAEqC       = a == c
#     print(f"whereAIsNonZero && whereAEqC=\n{(whereAIsNonZero & whereAEqC)}")
    
    bInheritedFromC = c*(whereAIsZero | (whereAIsNonZero & whereANeqC))
#     print(f"bInheritedFromC={bInheritedFromC}")
    
    bIsCOrZero        = whereAIsNonZero & whereAEqC
    cIsPlus           = c == +1
    cIsMinus          = c == -1
    numVaryingIndices = k = np.count_nonzero(bIsCOrZero)
    if k == 0:
        result = bInheritedFromC
        if len(result.shape) == 1:
            result = np.expand_dims(result, 0)
        return result
#     print(f"bIsCOrZero={bIsCOrZero}")
    insertPlusZero    = bIsCOrZero & cIsPlus
    insertMinusZero   = bIsCOrZero & cIsMinus
#     print(f"insertPlusZero={insertPlusZero}")
#     print(f"insertMinusZero={insertMinusZero}")
#     print(f"numVaryingIndices={k}")
    
    result = np.tile(bInheritedFromC, [2**k,1])
#     print(f"result=\n{result}")

    filler_columns = [plusZero if insertPlusZero[i] else minusZero 
                      for i in range(m) 
                      if insertPlusZero[i] or insertMinusZero[i]]
    filler_matrix  = cartesian_product(*filler_columns)# if not np.all(b == 0) else None # shape = (2^k, 1)
#     print(f"filler_matrix=\n{filler_matrix}")
    
    index_to_insert_column    = bIsCOrZero.nonzero()[0]
    index_of_column_to_insert = np.arange(filler_matrix.shape[-1])
    zipped                    = np.vstack([index_to_insert_column, 
                                           index_of_column_to_insert])
#     print(f"zipped=\n{zipped}")
    
    for each in zipped.T:
        index_for_insertion = each[0]
        index_to_insert     = each[1]
        result[:, index_for_insertion] = filler_matrix[:,index_to_insert]
#     print(f"result=\n{result}")
    
#     for b_prime in result:
#         actual_c = prague.priority_union(a, b_prime)
#         assert np.array_equal(actual_c, c), f"{a} + {b_prime} = {actual_c} ≠ {c}"
    
    if len(result.shape) == 1:
        result = np.expand_dims(result, 0)

    return result


def diff(c,a):
    '''
    Given pfvs c, a with meet m = c ∧ a,
    this calculates the b s.t.
        c - a = b
      ≡ c - m = b
    where c, a, and m are treated as partial relations on {feature labels} x {+1, 0,-1},
    '''
    m = meet_specification(c,a)
    b = c - m
    return b


def left_inv_priority_union(a, c, returnAsIntervalBounds=False):
    '''
    Let + denote right priority union, where for some unknown pfv b
      a + b = c
    
    If \ denotes 'left_inv_priority_union' = the left inverse of right 
    priority union, then 
      a \ c = ⊥ if there is any index i where c_i = 0 but a_i ≠ 0
      a \ c = { b | a + b = c }
    where a,b,c are all ternary pfvs.
    
    At the pointwise/ternary value level,
      x \ y, y < x = ⊥
      0 \ x        = {x}
      x \ x        = ↓x
      x \ y, y ≠ x = {y}
    In other words, 
      a_i = 0             -> b_i = c_i
      a_i ≠ 0 ∧ a_i ≠ c_i -> b_i = c_i
      a_i ≠ 0 ∧ a_i = c_i -> b_i = c_i ∨ 0

    By default this returns the stack of all solutions (if it exists). This set 
    of solutions will always form a bounded lattice - an interval of the 
    specification semilattice, in fact. If returnAsIntervalBounds=True, then 
    instead of returning the (potentially enormous) stack of solutions, the 
    bounds of the solution interval are returned as a tuple: 
        (max of solution interval, min of solution interval)
    '''
    whereClteA = (c == 0) & (a != 0)
    if np.any(whereClteA):
        return None
    glb        = meet_specification(a,c)
    k          = diff(c,glb)
    lub_interval = priority_union(glb,k)
    glb_interval = k
    if returnAsIntervalBounds:
        return (lub_interval, glb_interval)
    lcGlb      = lower_closure(glb)
    lcGlb_proj = priority_union(lcGlb,k)
    assert np.array_equal(max_of(lcGlb_proj), lub_interval), f"{max_of(lcGlb_proj)} vs. {lub_interval}"
    assert np.array_equal(min_of(lcGlb_proj), glb_interval), f"{min_of(lcGlb_proj)} vs. {glb_interval}"
    return lcGlb_proj


def right_inv_priority_union(c, b, returnAsIntervalBounds=False):
    '''
    Let + denote right priority union, where for some unknown pfv a
      a + b = c
    
    If / denotes 'right_inv_priority_union' = the right inverse of right 
    priority union, then 
      c / b = { a | a + b = c }
    where a,b,c are all ternary pfvs.
    
    At the pointwise/ternary value level,
      x / 0 = {x}
      x / x = ↑0
    In other words, the only case where '(c,b)' can be informative about 'a'
    is when 'b' is 0, in which case a=c.
    
    By default this returns the stack of all solutions (if it exists). This set 
    of solutions will always form a bounded meet semilattice - an upper closure 
    of the specification semilattice, in fact. If returnAsIntervalBounds=True, 
    then instead of returning the (potentially enormous) stack of solutions, the
    glb bound of the solution upper closure is returned simply as 
        min of solution upper closure

    '''
    if not lte_specification(b,c):
        return None
    bIsUndefinedMask = b == 0
    cMasked          = c * bIsUndefinedMask
    glb        = meet_specification(b,c)
    k          = diff(c,glb)
    offsetAt   = (c == 0) & (b == 0)
    offset     = np.ones(shape=c.shape, dtype=INT8) * offsetAt
    kOffset    = k + offset
    up_kOffset = upper_closure(kOffset)
    undoOffset = (-1 * offset) * offsetAt
    result     =   up_kOffset  + undoOffset

    assert np.array_equal(min_of(result),cMasked), f"min_of(result)={min_of(result)} vs. {cMasked}; c={c}, b={b}"
    assert np.array_equal(k, cMasked), f"{k} vs. {cMasked}; c={c}, b={b}"
    # assert np.array_equal(kOffset, cMasked), f"{kOffset} vs. {cMasked}; c={c}, b={b}"
    if returnAsIntervalBounds:
        return cMasked
    return result


def spe_update(a, b, object_inventory=None):
    '''
    Coerces a to reflect what's specified in b (per an SPE-style unconditioned
    rule analogous to "a ⟶ b"; does not support alpha notation).
    '''
    b_specification_mask   = np.abs(b, dtype=INT8)
    b_unspecification_mask = np.logical_not(b_specification_mask)
    prepped_a              = a * b_unspecification_mask
    new_a = prepped_a + b
    # new_a = a.copy()
    # for i in np.arange(new_a):
    #     if b[i] != 0:
    #         new_a[i] = b[i]
    return new_a
    # if object_inventory is None:
    #     return new_a

    # "If object_inventory is None, this interprets a and b literally; else, this
    # computes the set of minimal sets of changes you'd have to make to the
    # literal update to keep the result inside object_inventory, and returns those
    # resulting pfvs."

    # my_ext = extension(new_a, object_inventory)
#     if my_ext.sum() > 0:
#         return new_a

#     #TODO implement this efficiently
#     print(f'Coerced vector has empty extension:\n{new_a}')

#     #if new_a has an empty extension, then the only way to alter it and end up
#     #with something that has a nonempty extension is to pick some subset of
#     #currently specified indices (not including what's specified in b) and
#     #either unspecifying them or flipping their specification
#     new_a_specification_mask = np.abs(new_a, dtype=INT8)
#     new_a_mutatable_mask = new_a_specification_mask * b_unspecification_mask
#     mutable_indices = new_a_mutatable_mask.nonzero()[0]
#     l = mutable_indices.shape[0]
#     m = a.shape[0]
# #     print(f"m = {m}\n"
# #           f"a = {a}")

#     #For each of the l nonzero indices in new_a_mutable_mask, there are two
#     #possible updates, leading to a total of 2^l alterations of new_a.
#     #We want 'minimal' alterations with non-empty extension.
#     # basic_alterations = np.ones((2*l,m), dtype=INT8)
#     # for i in np.arange(2*l): #which basic alteration are we building?
#     #     j = i // 2 #which index are we using to build the basic alteration?
#     #     k = i % 2 #which effect is the basic alteration going to have?
#     #     mod_index = mutable_indices[j]
#     #     mod_type = {0:0,    #despecify
#     #                 1:-1}[k]#, #flip value
#     #                 # 2:-1}
#     #     basic_alterations[i, mod_index] = mod_type
#     # return basic_alterations
#     # alterations = np.ones((2**l,m), dtype=INT8)
#     # for i in np.arange(2**l):
#         # alterations
#     reachable_pfvs = []
#     k = 1
#     while k <= l:
# #         print(f'k={k}')
#         indices_to_modify = list(itertools.combinations(mutable_indices, k))
#         for index_set in indices_to_modify:
# #             print(f"index_set={index_set}")
#             modifications = list(itertools.product((0,-1), repeat=len(index_set)))
#             new_pfvs = np.tile(new_a, (len(modifications), 1))
# #             print(f"new_pfvs.shape = {new_pfvs.shape}")
#             for i, mod in enumerate(modifications):
#                 new_pfvs[i, np.array(index_set)] = np.array(mod)
# #                 print(f"\ti, mod = {i},{mod}\n"
# #                       f"\t{new_pfvs[i, np.array(index_set)]}")
#             new_exts = extensions(new_pfvs, object_inventory)
# #             print(f'new_exts =\n{new_exts}')
#             are_nonempty_indicator = new_exts.sum(axis=1) > 0
# #             print(f"are_nonempty_indicator =\n{are_nonempty_indicator}")
#             new_pfvs_with_nonempty_extension = new_pfvs[are_nonempty_indicator]
# #             print(f"new_pfvs_with_nonempty_extension =\n{new_pfvs_with_nonempty_extension}")
#             if new_pfvs_with_nonempty_extension.shape[0] > 0:
#                 reachable_pfvs.append(new_pfvs_with_nonempty_extension)
#         if len(reachable_pfvs) > 0:
#             return np.vstack(reachable_pfvs)
#         k+=1
#     raise Exception('wtf: no coercible vector within edit distance {k} that maintains {b}')


def make_rule(target, change):
    '''
    Returns a function that behaves like an unconditioned SPE-style rewrite
    rule 
      target → change
    '''
    assert target.shape[0] == change.shape[0], f"Shape mismatch: {target.shape} vs. {change.shape}"
    def phi(v):
        if lte_specification(target, v):
            return priority_union(v, change)
        else:
            return v
    return phi


def pseudolinear_inverse_possible(x,y):
    '''
    Calculates whether there exists a pair of vectors m, b s.t.
      x + m + b = y
    where 
     - '+' denotes right priority union. 
     - b only specifies indices unspecified in x.
     - m only specifies indices specified in x and specifies them to their 
       opposite value.
    
    This will be possible iff every feature index specified in x is also 
    specified in y, although x and y need not have the same specified VALUE.
    '''
    specified_in_x = np.abs(x) == 1
    specified_in_y = np.abs(y) == 1
    specified_in_y_masked_by_whats_specified_in_x = specified_in_y[specified_in_x]
    return np.all(specified_in_y_masked_by_whats_specified_in_x)


def pseudolinear_inverse(x,y):
    '''
    If there exists a pair of vectors m, b s.t.
      x + m + b = y
    where 
     - '+' denotes right priority union
     - b only specifies indices unspecified in x
     - m only specifies indices specified in x and specifies them to their 
       opposite value
    this returns those two vectors m,b.
    
    Otherwise this returns None.
    
    (See `pseudolinear_inverse_possible` for a description of the conditions 
    under which these two vectors exist.)
    '''
    if not pseudolinear_inverse_possible(x,y):
        return None
    
    specified_in_x    = np.abs(x) == 1
    unspecified_in_x  = np.abs(x) == 0
    specified_in_y    = np.abs(y) == 1
    specified_in_both = specified_in_y & specified_in_x
    differing         = x != y
    
    indices_of_y_that_are_in_b = specified_in_y & unspecified_in_x
    b = indices_of_y_that_are_in_b * y
    
    indices_of_y_that_are_in_m = specified_in_both & differing
    m = indices_of_y_that_are_in_m * y
    return (m,b)


def pseudolinear_decomposition(t,c):
    '''
    Returns the unique pair of vectors m,b s.t.
      x + m + b = x + c
    where
     - '+' denotes right priority union.
     - b only specifies indices unspecified in t.
     - m only specifies indices specified in t and species them to their 
       opposite value.
    '''
    specified_in_t    = np.abs(t) == 1
    unspecified_in_t  = np.abs(t) == 0
    specified_in_c    = np.abs(c) == 1
    specified_in_both = specified_in_t & specified_in_c
    differing         = t != c
    
    indices_of_c_that_are_in_b = specified_in_c & unspecified_in_t
    b                          = indices_of_c_that_are_in_b * c
    
    indices_of_c_that_are_in_m = specified_in_both & differing
    m                          = indices_of_c_that_are_in_m * c
    

    lte = lte_specification
    assert np.array_equal(c, priority_union(m,b)) | (lte(c,t) & (m.sum() == 0 & b.sum() == 0)), f"{t}→{c} ≠ {t}→{m}→{b} (= {t}→{priority_union(m,b)})"
    
    return (m,b)


#############################
# SPECIFICATION SEMILATTICE #
#############################

# Slightly redundant with the next function, but also effectively documentation
# for it, since the next function has a different type signature and is a bit
# much to grasp all at once.
def lte_specification(u, v):
    '''Given two partial feature vectors u, v, this calculates whether
      u ≤ v
    in the specification semilattice.

    Intuitively, u ≤ v holds iff u is *less specified* than v. For example,
        [+nasal]
    is less specified than [+nasal, +velar], as is
        [+velar]
    but [-nasal] is incomparable. (We're not measuring *degree* of
    specification here.)

    Given two partial feature vectors u, v both with m ternary features, let
      u[i], v[i]
    denote the ith feature value for u and v, respectively.

    At the element-level
        u[i] ≤ v[i]
    iff
        (u[i] == v[i]) or (u[i] == 0)
    i.e.
      +1 ≤ +1
      -1 ≤ -1
       0 ≤  0
       0 ≤ +1
       0 ≤ -1
    in this semilattice ordering, and and this ordering on feature values is
    extended to vectors in the natural way: this function returns
        1 iff u[i] ≤ v[i] for all i ∈ [0, m-1]
    and
        0 otherwise.
    '''
    return ((u == v) | (u == 0)).all()


def lte_specification_stack_left(M, u, axis=1):
    '''Given a partial feature vector u and a matrix (stack) of partial feature
    vectors M (one vector per row), this efficiently calculates whether
        M[i] ≤ u
    in the specification semilattice for each vector i. In other words, this
    checks membership of M[i] in the lower closure of u.

    If
        u.shape == (m,)
    and
        M.shape == (k,m)
    then
        lte_specification_stack_left(M,u).shape == (k,)
    '''
    return (np.equal(u, M) | np.equal(M, 0)).prod(axis=axis, dtype=INT8)


def lte_specification_stack_right(u, M, axis=1):
    '''Given a partial feature vector u and a matrix (stack) of partial feature
    vectors M (one vector per row), this efficiently calculates whether
        u ≤ M[i]
    in the specification semilattice for each vector i. In other words, this
    checks membership of each M[i] in the upper closure of u.

    If
        u.shape == (m,)
    and
        M.shape == (k,m)
    then
        lte_specification_stack_right(u,M).shape == (k,)
    '''
    return (np.equal(M, u) | np.equal(u, 0)).prod(axis=axis, dtype=INT8)


def lte_specification_dagwood(M, U, axis=2):
    '''
    Given two stacks of partial feature vectors M::(l,m), U::(o,m),
    this returns a matrix R::(l,o) where
        R[i,j] == 1  iff  M[i] ≤ U[j]

    In other words, this is the natural 'stack' version of
        lte_specification_stack_right
    or equivalently, an efficient version of
        R = np.array([[prague.lte(row, col) for col in U]
                      for row in M], dtype=np.int8)
    '''
    cart_prod_arr = np.array(cartesian_product_stack(M,U), dtype=INT8)
    result        = (np.equal(cart_prod_arr[0,:,:],cart_prod_arr[1,:,:]) | np.equal(cart_prod_arr[0,:,:], 0)).prod(axis=2, dtype=INT8)
    return result


def meet_specification(u=None, v=None, M=None):
    '''Given two partial feature vectors u,v, returns the unique single partial
    feature vector that is the greatest lower bound of u and v in the
    specification semilattice. This will be a vector with every specified value
    that is specified with exactly the same value in both u and v, and no other
    specified values.

    Alternately: given a stack of partial feature vectors M (one vector per
    row), returns the greatest lower bound of the stack in the specification
    semilattice.
    '''
    if u is not None and v is not None:
        return np.sign(  np.equal(u, v) * (u + v), dtype=INT8)
    elif M is not None:
        meet_mask = np.all(M == M[0,:], axis=0)
        return meet_mask * M[0]
        # return np.sign( np.equal.reduce(M, axis=0) * np.sum(M, axis=0), dtype=INT8)
    else:
        raise Exception('Provide exactly two vectors u,v or else a stack M.')


def specification_degree(u=None, M=None):
    '''
    Given a pfv u or a stack of pfvs M, calculate the degree(s) of 
    specification.
    
    Ex: If
     a = np.array([-1,0,1], dtype=np.int8)
     b = np.array([-1,0,0], dtype=np.int8)
     c = np.array([a,b], dtype=np.int8)
    then
     specification_degree(a) == 2
     specification_degree(b) == 1
     specifcation_degree(M=c) == np.array([2,1])
    '''
    if u is not None:
        return np.abs(u).sum()
    elif M is not None:
        return np.abs(M).sum(axis=-1)
    else:
        raise Exception("Provide a vector as argument u or a stack of vectors as M.")

def minimally_specified(M):
    '''
    Given a stack of pfvs M, returns the pfvs whose specification is minimal 
    relative to other pfvs in M.
    
    Ex: If
     a = np.array([-1,0,1], dtype=np.int8)
     b = np.array([-1,0,0], dtype=np.int8)
     c = np.array([a,b], dtype=np.int8)
    then
     minimally_specified(c) == np.array([[-1,0,0]])
    '''
    specs    = specification_degree(M=M)
    spec_min = np.min(specs)
    return M[specs == spec_min]


def maximally_specified(M):
    '''
    Given a stack of pfvs M, returns the pfvs whose specification is maximal 
    relative to other pfvs in M.
    
    Ex: If
     a = np.array([-1,0,1], dtype=np.int8)
     b = np.array([-1,0,0], dtype=np.int8)
     c = np.array([a,b], dtype=np.int8)
    then
     maximally_specified(c) == np.array([[-1,0,1]])
    '''
    specs    = specification_degree(M=M)
    spec_max = np.max(specs)
    return M[specs == spec_max]


def min_of(M):
    '''
    Given a k x m stack M of m k-length pfvs, returns 
        the v ∈ M s.t. v is the greatest lower bound of M
    if such a v exists, otherwise returns None.
    '''
    # FIXME test
    glb = meet_specification(M=M)
    Ms = stack_to_set(M)
    if HashableArray(glb) in Ms:
        return glb
    else:
        return None


def max_of(M):
    '''
    Given a k x m stack M of m k-length pfvs, returns 
        the v ∈ M s.t. v is the least upper bound of M
    if such a v exists, otherwise returns None.
    '''
    # FIXME test
    lub_exists = join_specification_possible_stack(M)
    if not lub_exists:
        return None
    lub = join_specification_stack(M=M)
    Ms = stack_to_set(M)
    if HashableArray(lub) in Ms:
        return lub
    else:
        return None


def contains_own_meet(M):
    '''
    Given a stack of pfvs M, this returns whether M contains its own greatest 
    lower bound.
    '''
    return min_of(M) is not None


def contains_own_join(M):
    '''
    Given a stack of pfvs M, this returns whether M contains its own least 
    upper bound.
    '''
    return max_of(M) is not None


def closure_under_meet(M):
    '''
    Given a k x m stack M of k m-length pfvs, returns a stack representing
    the closure of M under meet.
    '''
    Ms = stack_to_set(M)
    meets = {HashableArray(meet_specification(a.unwrap(), b.unwrap())) 
             for a in Ms for b in Ms}
    return hashableArrays_to_stack(meets)


def is_closed_under_meet(M):
    '''
    Given a k x m stack M of k m-length pfvs, returns whether M is closed 
    under meet.
    '''
    Ms = stack_to_set(M)
    meets = {HashableArray(meet_specification(a.unwrap(), b.unwrap())) 
             for a in Ms for b in Ms}
    in_closure_but_not_in_Ms = meets - Ms
    return len(in_closure_but_not_in_Ms) == 0


def is_meet_total_over(M):
    '''
    Given a k x m stack M of k m-length vectors, returns whether meet is total 
    over M.

    NOTE: this will *always* be true for a stack of well-formed pfvs.
    '''
    return True


def closure_under_join(M):
    '''
    Given a k x m stack M of k m-length pfvs, returns a stack representing
    the closure of M under join (wrt all pairs of M where join is defined).
    '''
    Ms = stack_to_set(M)
    joinablePairs = {(a,b) for a in Ms for b in Ms 
                     if join_specification_possible(a.unwrap(), b.unwrap())}
    joins = {HashableArray(join_specification(a.unwrap(), b.unwrap())) 
             for (a,b) in joinablePairs}
    return hashableArrays_to_stack(joins)


def is_join_total_over(M):
    '''
    Given a k x m stack M of k m-length vectors, returns whether join is total 
    over M.
    '''
    Ms = stack_to_set(M)
    allPairs = {(a,b) for a in Ms for b in Ms}
    joinablePairs = {(a,b) for (a,b) in allPairs
                     if join_specification_possible(a.unwrap(), b.unwrap())}
    return allPairs == joinablePairs


def is_closed_under_join(M):
    '''
    Given a k x m stack M of k m-length pfvs, returns whether M is closed 
    under join (wrt all pairs of M where join is defined).
    '''
    Ms = stack_to_set(M)
    joinablePairs = {(a,b) for a in Ms for b in Ms 
                     if join_specification_possible(a.unwrap(), b.unwrap())}
    joins = {HashableArray(join_specification(a.unwrap(), b.unwrap())) 
             for (a,b) in joinablePairs}
    in_closure_but_not_in_Ms = joins - Ms
    return len(in_closure_but_not_in_Ms) == 0


def join_naive(v,u):
    def j(v_i,u_i):
        if v_i == 0:
            return u_i
        elif u_i == 0:
            return v_i
        elif v_i == u_i:
            return v_i
        else:
            return 9
    return np.array([j(v_i,u_i) for v_i,u_i in zip(v,u)], dtype=np.int8)


def join_specification_possible_stack(M):
    # column_is_specified_somewhere = np.any(M, axis=0)
    column_is_all_gte_zero = np.all(M >= 0, axis=0)
    column_is_all_lte_zero = np.all(M <= 0, axis=0)
    return np.all(column_is_all_gte_zero | column_is_all_lte_zero)


def join_specification_stack(M):
    if not join_specification_possible_stack(M):
        return None
    
    column_is_all_gte_zero = np.all(M >= 0, axis=0)
    column_is_all_lte_zero = np.all(M <= 0, axis=0)
    # column_is_all_zero     = np.all(M == 0, axis=0)

    m = M.shape[-1]
    plus  = make_ones_pfv(m)
    minus = -1 * plus
    # zero  = make_zero_pfv(m)

    j = (column_is_all_lte_zero * minus) + (column_is_all_gte_zero * plus)
    return j


def join_specification_possible(u=None, v=None, M=None):
    '''
    Given two partial feature vectors u,v, returns whether their join in
    the specification semilattice exists.

    Alternately, given a stack of vectors M, returns whether a join of the stack
    in the specification semilattice exists.
    '''
    if u is not None and v is not None:
        specified_in_u    = np.abs(u) == 1
        specified_in_v    = np.abs(v) == 1
        specified_in_both = specified_in_u & specified_in_v
        
        same           = u == v
        different      = u != v
        
        incompatible   = specified_in_both & different
        return not np.any(incompatible)
    elif M is not None:
        return join_specification_possible_stack(M)
    else:
        raise Exception('Provide exactly two vectors u,v or else a stack M.')


def join_specification(u=None, v=None, M=None):
    '''Given two partial feature vectors u,v, returns the unique single partial
    feature vector that is the least upper bound of u and v in the
    specification semilattice, if it exists. This will be a vector with every 
    specified value that is specified in u and with every specified value that 
    is specified in v, and with no other specified values.
    
    Alternately: given a stack of partial feature vectors M (one vector per
    row), returns the least upper bound of the stack in the specification
    semilattice if it exists, else None.

    If no such join exists, returns None.
    '''
    if u is not None and v is not None:
        specified_in_u    = np.abs(u) == 1
        specified_in_v    = np.abs(v) == 1
        specified_in_both = specified_in_u & specified_in_v
        
        same           = u == v
        different      = u != v
        
        incompatible   = specified_in_both & different
        if np.any(incompatible):
            return None
        
        specified_just_in_u = specified_in_u & ~specified_in_both
        specified_just_in_v = specified_in_v & ~specified_in_both
        
        join = (specified_in_both * u) + (specified_just_in_u * u) + (specified_just_in_v * v)
        return join
    elif M is not None:
        return join_specification_stack(M)
    else:
        raise Exception('Provide exactly two vectors u,v or else a stack M.')


def is_meet_semilattice(M):
    return is_meet_total_over(M) and is_closed_under_meet(M)


def is_join_semilattice(M):
    Ms = stack_to_set(M)
    allPairs = {(a,b) for a in Ms for b in Ms}
    joinablePairs = {(a,b) for (a,b) in allPairs
                     if join_specification_possible(a.unwrap(), b.unwrap())}
    if allPairs != joinablePairs:
        return False
    joins = {HashableArray(join_specification(a.unwrap(), b.unwrap())) 
             for (a,b) in joinablePairs}
    in_closure_but_not_in_Ms = joins - Ms
    return len(in_closure_but_not_in_Ms) == 0


def is_lattice(M):
    return is_join_semilattice(M) and is_meet_semilattice(M)


def is_bounded_meet_semilattice(M):
    return is_meet_semilattice(M) and contains_own_meet(M)


def is_bounded_join_semilattice(M):
    return is_join_semilattice(M) and contains_own_join(M)


def is_bounded_lattice(M):
    return is_lattice(M) and contains_own_join(M) and contains_own_meet(M)


def is_lower_closure(M):
    lub_exists = join_specification_possible_stack(M)
    if not lub_exists:
        return False
    lub = join_specification_stack(M=M)
    Ms = stack_to_set(M)
    if not HashableArray(lub) in Ms:
        return False
    lub_lc = lower_closure(lub)
    lub_lcs = stack_to_set(lub_lc)
    return lub_lcs == Ms


def is_upper_closure(M):
    glb = meet_specification(M=M)
    Ms = stack_to_set(M)
    if not HashableArray(glb) in Ms:
        return False
    glb_uc = upper_closure(glb)
    glb_ucs = stack_to_set(glb_uc)
    return glb_ucs == Ms


def dual(u):
    '''
    An involution that returns a new pfv where every specified feature of u is 
    flipped to its opposite value.
    '''
    return -1 * u


def normalize(t,c):
    '''
    Given a target vector-change vector pair representing an unconditioned 
    rewrite rule, this returns (t,c') where c' is distinct from c just in case 
     - c contains feature specifications that are redundant with respect to t 
       (≡ c ∧ t ≠ 0^m)
       - e.g. in [+ + 0] -> [0 + -], the medial + of the change vector does 
         nothing and this rule is equivalent to [+ + 0] -> [0 0 -], so 
         ([+ + 0], [0 0 -]) is what would be returned.
       - c' = the minimal element in (c ∧ t) \ c
         where a \ c = {b | a + b = c}
           - treating (c ∧ t) and c as partial relations, c' will be c - (c ∧ t)
         where a + b = the right priority union of a and b
    '''
    meet_tc = meet_specification(t,c)
    if np.abs(meet_tc).sum() == 0:
        return (t,c)
    specified_in_t = np.abs(t) == 1
    specified_in_c = np.abs(c) == 1
    same           = t == c
    zero_out_in_c  = specified_in_c & same
    keep_from_c    = ~zero_out_in_c
    
    assert np.any(zero_out_in_c), f"Expected to zero out some indices of c given (t,c)=({t},{c}) but zero-out mask = {zero_out_in_c}"
    
    c_prime        = keep_from_c * c
    return (t, c_prime)


def is_antichain(M):
    '''
    Returns True iff the set of pfvs in the k x m stack M are all pairwise 
    incomparable (i.e. the set of pfvs in M form an anti-chain).
    '''
    Ms = stack_to_set(M)
    allPairs = {(a,b) for a in Ms for b in Ms}
    for aWrapped, bWrapped in allPairs:
        a, b  = aWrapped.unwrap(),bWrapped.unwrap()
        aLTEb = lte_specification(a,b)
        bLTEa = lte_specification(b,a)
        comparable = aLTEb or bLTEa
        if comparable and not np.array_equal(a,b):
            return False
    return True


def is_meet_semilattice_distributive(sl_stack, returnCounterexamples=False):
    '''
    Given a set M of pfvs that form a meet semilattice, M is a *distributive* iff
    for all a, b, x
     - If a ∧ b ≤ x then ∃ a', b' . a ≤ a', b ≤ b' and x = a' ∧ b'.
    
    By default this returns whether this property holds of the parameter stack
    sl_stack. If returnCounterexamples is True, this instead returns the set of 
    counterexamples found.
    '''
    assert is_meet_semilattice(sl_stack), f"stack must form meet semilattice:\n{sl_stack}"
    Ms = stack_to_set(sl_stack)
    counterexamples = set()
    allTriples = {(a,b,x) for a in Ms for b in Ms for x in Ms}
    for aWrapped, bWrapped, xWrapped in allTriples:
        a, b, x = aWrapped.unwrap(), bWrapped.unwrap(), xWrapped.unwrap()
        m = meet_specification(a,b)
        if lte_specification(m, x):
            ucx  = upper_closure(x)
            ucxS = stack_to_set(ucx)
            allPairs  = {(aPrime, bPrime) for aPrime in ucxS for bPrime in ucxS}
            solutions = set()
            for aPrimeWrapped, bPrimeWrapped in allPairs:
                aPrime, bPrime = aPrimeWrapped.unwrap(), bPrimeWrapped.unwrap()
                aLTEaPrime     = lte_specification(a, aPrime)
                bLTEbPrime     = lte_specification(b, bPrime)
                xIsMeetAPrimeBprime = np.array_equal(x, meet_specification(aPrime, bPrime))
                if aLTEaPrime and bLTEbPrime and xIsMeetAPrimeBprime:
                    solutions.add((aPrimeWrapped, bPrimeWrapped))
                if len(solutions) > 0:
                    break
            if len(solutions) == 0:
                counterexamples.add((aWrapped,bWrapped,xWrapped))
    if returnCounterexamples:
        return counterexamples
    return len(counterexamples) == 0


def is_join_semilattice_distributive(sl_stack, returnCounterexamples=False):
    '''
    Given a set M of pfvs that form a join semilattice, M is a *distributive* iff
    for all a, b, x
     - If x ≤ a ∧ b then ∃ a', b' . a' ≤ a, b' ≤ b and x = a' ∨ b'.
    
    By default this returns whether this property holds of the parameter stack
    sl_stack. If returnCounterexamples is True, this instead returns the set of 
    counterexamples found.
    '''
    assert is_join_semilattice(sl_stack), f"stack must form join semilattice:\n{sl_stack}"
    Ms = stack_to_set(sl_stack)
    counterexamples = set()
    allTriples = {(a,b,x) for a in Ms for b in Ms for x in Ms}
    for aWrapped, bWrapped, xWrapped in allTriples:
        a, b, x = aWrapped.unwrap(), bWrapped.unwrap(), xWrapped.unwrap()
        j = join_specification(a,b)
        if j is not None:
            if lte_specification(x, j):
                lcx  = lower_closure(x)
                lcxS = stack_to_set(lcx)
                allPairs  = {(aPrime, bPrime) for aPrime in lcxS for bPrime in lcxS}
                solutions = set()
                for aPrimeWrapped, bPrimeWrapped in allPairs:
                    aPrime, bPrime   = aPrimeWrapped.unwrap(), bPrimeWrapped.unwrap()
                    aPrimeLTEa       = lte_specification(aPrime, a)
                    bPrimeLTEb       = lte_specification(bPrime, b)
                    aPrimeJoinbPrime = join_specification(aPrime, bPrime)
                    if aPrimeJoinbPrime is not None:
                        xIsJoinAPrimeBprime = np.array_equal(x, aPrimeJoinbPrime)
                        if aPrimeLTEa and bPrimeLTEb and xIsJoinAPrimeBprime:
                            solutions.add((aPrimeWrapped, bPrimeWrapped))
                    if len(solutions) > 0:
                        break
                if len(solutions) == 0:
                    counterexamples.add((aWrapped,bWrapped,xWrapped))
        else:
            raise Exception("stack must form join semilattice, but join is undefined: {a} ∨ {b} dne\nstack:{sl_stack}")
    if returnCounterexamples:
        return counterexamples
    return len(counterexamples) == 0


def is_lattice_distributive(l_stack, returnCounterexamples=False):
    '''
    A lattice is distributive iff meet always distributes over join:
        a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)
    
    Given a stack of pfvs that are a lattice, this (by default) indicates 
    whether the lattice is distributive. If returnCounterexamples=True, then
    this returns the (possibly empty) set of counterexamples found.

    NB distributivity of a lattice implies modularity.
    '''
    assert is_lattice(l_stack), f"stack must form lattice:\n{l_stack}"
    Ms = stack_to_set(l_stack)
    allTriples = {(a,b,c) for a in Ms for b in Ms for c in Ms}
    counterexamples = set()
    for aWrapped, bWrapped, cWrapped in allTriples:
        a,b,c   = aWrapped.unwrap(), bWrapped.unwrap(), cWrapped.unwrap()
        bJc     = join_specification(b,c)
        aMbJc   = meet_specification(a,bJc)
        aMb     = meet_specification(a,b)
        aMc     = meet_specification(a,c)
        aMbJaMc = join_specification(aMb, aMc)
        if not np.array_equal(aMbJc, aMbJaMc):
            counterexamples.add((aWrapped,bWrapped,cWrapped, f"{a} ∧ ({b} ∨ {c}) = {a} ∧ {bJc} = {aMbJc} ≠ {aMbJaMc} = {aMb} ∨ {aMc} = ({a} ∧ {b}) ∨ ({a} ∧ {c})"))
    if returnCounterexamples:
        return counterexamples
    return len(counterexamples) == 0


def is_lattice_modular(l_stack, returnCounterexamples=False):
    '''
    A lattice is modular iff the following associativity-like property holds:
        If a ≤ b, then ∀x, a ∨ (x ∧ b) = (a ∨ x) ∧ b
    
    Given a stack of pfvs that are a lattice, this (by default) indicates 
    whether the lattice is modular. If returnCounterexamples=True, then
    this returns the (possibly empty) set of counterexamples found.
    '''
    assert is_lattice(l_stack), f"stack must form lattice:\n{l_stack}"
    Ms = stack_to_set(l_stack)
    allPairs   = {(a,b) for a in Ms for b in Ms}
    counterexamples = set()
    for aWrapped, bWrapped in allPairs:
        a,b = aWrapped.unwrap(), bWrapped.unwrap()
        if lte_specification(a,b):
            for xWrapped in Ms:
                x      = xWrapped.unwrap()
                xMb    = meet_specification(  x,   b)
                aJx    = join_specification(  a,   x)
                aJ_xMb = join_specification(  a, xMb)
                aJx_Mb = meet_specification(aJx,   b)
                if not np.array_equal(aJ_xMb, aJx_Mb):
                    counterexamples.add((aWrapped,bWrapped,xWrapped, f"{a} ∨ ({x} ∧ {b}) = {a} ∨ {xMb} = {aJ_xMb} ≠ {aJx_Mb} = {aJx} ∧ {b} = ({a} ∨ {x}) ∧ {b}"))
    if returnCounterexamples:
        return counterexamples
    return len(counterexamples) == 0


def complement_search(x, l_stack):
    '''
    Given 
     - a pfv x 
     - a stack of pfvs that form a bounded lattice and contain x
    this returns a stack C of pfvs such that ∀c ∈ C
     - a ∨ c = join(lattice)
     - a ∧ c = meet(lattice)
    '''
    assert is_bounded_lattice(l_stack), f"stack must form bounded lattice:\n{l_stack}"
    top = join_specification_stack(l_stack)
    bot = meet_specification(M=l_stack)
    Ms  = stack_to_set(l_stack)
    Cs = {c for c in Ms 
          if np.array_equal(join_specification(x,c.unwrap()), top) and
             np.array_equal(meet_specification(x,c.unwrap()), bot)}
    C = hashableArrays_to_stack(Cs)
    return C


def complement_exact(x, l_stack):
    '''
    Given 
     - a pfv x 
     - a stack of pfvs that form a bounded lattice and contain x
    this returns the c in the lattice such that both
     - a ∨ c = join(lattice)
     - a ∧ c = meet(lattice)
    '''
    assert is_bounded_lattice(l_stack), f"stack must form bounded lattice:\n{l_stack}"
    top = join_specification_stack(l_stack)
    bot = meet_specification(M=l_stack)
    justMin = diff(top,x)
    xJoinMin = join_specification(x,justMin)
    if xJoinMin is None:
        raise Exception(f"stack must form a lattice, but join was undefined:\n{x} ∨ {justMin} dne\nstack:{l_stack}")
    xMeetMin = meet_specification(x,justMin)
    assert np.array_equal(xJoinMin,top), f"{x}, {justMin} | {xJoinMin} ≠ {top}"
    assert np.array_equal(xMeetMin,bot), f"{x}, {justMin} | {xMeetMin} ≠ {bot}"
    return justMin


def is_complemented_lattice(l_stack, returnCounterexamples=False):
    '''
    Given a stack M of pfvs forming a bounded lattice, M is a *complemented* 
    lattice iff ∀x ∈ M, ∃c ∈ M s.t.
     - x ∨ c = join(M)
     - x ∧ c = meet(M)
    c need not in general be unique, though in distributive lattices complements
    will be unique (if one exists for any given element).

    By default, this returns whether l_stack is a complemented lattice. If 
    returnCounterexamples=True, this returns the (possibly empty) set of 
    counterexamples instead.
    '''
    assert is_bounded_lattice(l_stack), f"stack must form bounded lattice:\n{l_stack}"
    Ms = stack_to_set(l_stack)
    counterexamples = set()
    for xWrapped in Ms:
        x      = xWrapped.unwrap()
        cStack = complement_search(x, l_stack)
        cSet   = stack_to_set(cStack)
        if len(cSet) == 0:
            counterexamples.add(xWrapped)
    if returnCounterexamples:
        return counterexamples
    return len(counterexamples) == 0


def distribution_un_bin(un_op_f, bin_op_g, M, eq=None, returnCounterexamples=False):
    '''
    Given
     - a unary function f: V → V
     - a binary function g: V → V
     - a stack of pfvs M
     - an optional equality function for comparing function outputs
    this checks whether
        f(g(a,b)) = g(f(a), f(b)) ≡ "f distributes over g"
    holds for all a,b,c in M.
    
    By default this returns a boolean. If returnCounterexamples=True, it returns
    the (possibly empty) set of counterexamples found instead.

    NB If f or g are partial, then any a,b where 
      - f(g(a,b)) exists but g(f(a), f(b)) does not (or vice versa)
    is considered a counterexample to f distributing over g.
    '''
    if eq is None:
        eq = np.array_equal
    Ms = stack_to_set(M)
    allPairs = {(a,b) for a in Ms for b in Ms}
    counterexamples = set()
    for aWrapped, bWrapped in allPairs:
        reasons = []
        a,b     = aWrapped.unwrap(), bWrapped.unwrap()
        gab     = bin_op_g(a,b)
        fa      = un_op_f(a)
        fb      = un_op_f(b)

        reasons_gfafb_is_not_calculable = []
        if fa is None:
            reasons_gfafb_is_not_calculable.append(f"f({a}) dne")
        if fb is None:
            reasons_gfafb_is_not_calculable.append(f"f({b}) dne")
        if fa is not None and fb is not None:
            gfafb = bin_op_g(fa, fb)
            if gfafb is None:
                reasons_gfafb_is_not_calculable.append(f"g(f({a}),f({b})) dne")
        
        reasons_fgab_is_not_calculable = []
        if gab is None:
            reasons_fgab_is_not_calculable.append(f"g({a},{b}) dne")
        else:
            fgab = un_op_f(gab)
            if fgab is None:
                reasons_fgab_is_not_calculable.append(f"f(g({a},{b})) dne")
        
        if len(reasons_gfafb_is_not_calculable) == 0 and len(reasons_fgab_is_not_calculable) > 0:
            assert gfafb is not None
            assert fgab  is     None
            reasons.append(f"g(f({a},f({b})) exists but f(g({a},{b})) dne")
            reasons.extend(reasons_fgab_is_not_calculable)
            counterexamples.add((aWrapped, bWrapped, tuple(reasons)))
        elif len(reasons_gfafb_is_not_calculable) > 0 and len(reasons_fgab_is_not_calculable) == 0:
            assert gfafb is     None
            assert fgab  is not None
            reasons.append(f"f(g({a},{b})) exists but g(f({a},f({b})) dne")
            reasons.extend(reasons_gfafb_is_not_calculable)
            counterexamples.add((aWrapped, bWrapped, tuple(reasons)))
        elif len(reasons_gfafb_is_not_calculable) == 0 and len(reasons_fgab_is_not_calculable) == 0:
            assert gfafb is not None
            assert fgab  is not None
            if not eq(gfafb, fgab):
                counterexamples.add((aWrapped,bWrapped, f"f(g({a},{b})) = f({gab}) = {fgab} ≠ {gfafb} = g({fa},{fb}) = g(f({a}), f({b}))"))
        else:
            pass
    if returnCounterexamples:
        return counterexamples
    return len(counterexamples) == 0


def left_distribution_bin_bin(bin_op_f, bin_op_g, M, eq=None, returnCounterexamples=False):
    '''
    Given 
      - two binary functions f,g: V -> V on pfvs
      - a stack of pfvs M
      - an optional equality function for comparing function outputs
    this checks whether 
        f(a, g(b,c)) = g(f(a,b), f(a,c)) ≡ "f distributes over g from the left"
    holds for all a,b,c in M.
    
    By default this returns a boolean. If returnCounterexamples=True, it returns
    the (possibly empty) set of counterexamples found instead.

    NB If f or g are partial, then any a,b,c where 
      - f(a, g(b,c)) exists but g(f(a,b), f(a,c)) does not (or vice versa)
    is considered a counterexample to f distributing over g from the left.
    '''
    if eq is None:
        eq = np.array_equal
    Ms = stack_to_set(M)
    allTriples = {(a,b,c) for a in Ms for b in Ms for c in Ms}
    counterexamples = set()
    for aWrapped, bWrapped, cWrapped in allTriples:
        a,b,c   = aWrapped.unwrap(), bWrapped.unwrap(), cWrapped.unwrap()
        reasons = []
        bGc     = bin_op_g(b,c)
        aFb     = bin_op_f(a,b)
        aFc     = bin_op_f(a,c)
        aFbgc   = bin_op_f(a,bGc) if bGc is not None else None
        afbGafc = bin_op_g(aFb, aFc) if (aFb is not None) and (aFc is not None) else None
        
        reasons_aFbgc_is_not_calculable = []
        if bGc is None:
            reasons_aFbgc_is_not_calculable.append(f"g({b},{c}) dne")
        else:
            if aFbgc is None:
                reasons_aFbgc_is_not_calculable.append(f"g({b},{c}) exists but f({a},g({b},{c})) dne")

        reasons_afbGafc_is_not_calculable = []
        if aFb is None:
            reasons_afbGafc_is_not_calculable.append(f"f({a},{b}) dne")
        if aFc is None:
            reasons_afbGafc_is_not_calculable.append(f"f({a},{c}) dne")
        if (aFb is not None) and (aFc is not None) and (afbGafc is None):
            reasons_afbGafc_is_not_calculable.append(f"f({a},{b}), f({a},{c}) exist but g(f({a},{b}), f({a},{c})) dne")
        
        if len(reasons_aFbgc_is_not_calculable) > 0 and len(reasons_afbGafc_is_not_calculable) == 0:
            assert aFbgc   is     None
            assert afbGafc is not None
            reasons.append(f"g(f({a},{b}), f({a},{c})) exists but f({a},g({b},{c})) dne")
            reasons.extend(reasons_aFbgc_is_not_calculable)
            counterexamples.add((aWrapped,bWrapped,cWrapped,tuple(reasons)))
        elif len(reasons_aFbgc_is_not_calculable) == 0 and len(reasons_afbGafc_is_not_calculable) > 0:
            assert aFbgc   is not None
            assert afbGafc is     None
            reasons.append(f"f({a},g({b},{c})) exists but g(f({a},{b}), f({a},{c})) dne")
            reasons.extend(reasons_afbGafc_is_not_calculable)
            counterexamples.add((aWrapped,bWrapped,cWrapped,tuple(reasons)))
        elif len(reasons_aFbgc_is_not_calculable) == 0 and len(reasons_afbGafc_is_not_calculable) == 0:
            assert aFbgc   is not None
            assert afbGafc is not None
            if not eq(aFbgc, afbGafc):
                counterexamples.add((aWrapped,bWrapped,cWrapped, f"f({a},g({b},{c})) = f({a}, {bGc}) = {aFbgc} ≠ {afbGafc} = g({aFb}, {aFc}) = g(f({a},{b}),f({a},{c}))"))
        else:
            pass
    if returnCounterexamples:
        return counterexamples
    return len(counterexamples) == 0


def right_distribution_bin_bin(bin_op_f, bin_op_g, M, eq=None, returnCounterexamples=False):
    '''
    Given 
      - two binary functions f,g on pfvs with the same domain and codomain
      - a stack of pfvs M
      - an optional equality function for comparing function outputs
    this checks whether 
        f(g(b,c), a) = g(f(b,a), f(c,a)) ≡ "f distributes over g from the right"
    holds for all a,b,c in M.
    
    By default this returns a boolean. If returnCounterexamples=True, it returns
    the (possibly empty) set of counterexamples found instead.

    NB If f or g are partial, then any a,b,c where 
      - f(g(b,c), a) exists but g(f(b,a), f(c,a)) does not (or vice versa)
    is considered a counterexample to f distributing over g from the right.
    '''
    if eq is None:
        eq = np.array_equal
    Ms = stack_to_set(M)
    allTriples = {(a,b,c) for a in Ms for b in Ms for c in Ms}
    counterexamples = set()
    for aWrapped, bWrapped, cWrapped in allTriples:
        reasons = []
        a,b,c   = aWrapped.unwrap(), bWrapped.unwrap(), cWrapped.unwrap()
        bGc     = bin_op_g(b,c)
        bFa     = bin_op_f(b,a)
        cFa     = bin_op_f(c,a)
        bgcFa   = bin_op_f(bGc, a) if bGc is not None else None
        bfaGcfa = bin_op_g(bFa, cFa) if (bFa is not None and cFa is not None) else None

        reasons_bgcFa_is_not_calculable = []
        if bGc is None:
            reasons_bgcFa_is_not_calculable.append(f"g({b},{c}) dne")
        elif bgcFa is None:
            reasons_bgcFa_is_not_calculable.append(f"g({b},{c}) exists but f(g({b},{c}),{a}) dne")

        reasons_bfaGcfa_is_not_calculable = []
        if bFa is None:
            reasons_bfaGcfa_is_not_calculable.append(f"f({b},{a}) dne")
        if cFa is None:
            reasons_bfaGcfa_is_not_calculable.append(f"f({c},{a}) dne")
        if (bFa is not None) and (cFa is not None) and (bfaGcfa is None):
            reasons_bfaGcfa_is_not_calculable.append(f"f({b},{a}), f({c},{a}) exist but g(f({b},{a}),f({c},{a})) dne")

        if len(reasons_bgcFa_is_not_calculable) == 0 and len(reasons_bfaGcfa_is_not_calculable) > 0:
            assert bgcFa   is not None
            assert bfaGcfa is     None
            reasons.append(f"f(g({b},{c}),{a}) exists but g(f({b},{a}),f({c},{a})) dne")
            reasons.extend(reasons_bfaGcfa_is_not_calculable)
            counterexamples.add((aWrapped,bWrapped,cWrapped, tuple(reasons)))
        elif len(reasons_bgcFa_is_not_calculable) > 0 and len(reasons_bfaGcfa_is_not_calculable) == 0:
            assert bgcFa   is     None
            assert bfaGcfa is not None
            reasons.append(f"g(f({b},{a}),f({c},{a})) exists but f(g({b},{c}),{a}) dne")
            reasons.extend(reasons_bgcFa_is_not_calculable)
            counterexamples.add((aWrapped,bWrapped,cWrapped, tuple(reasons)))
        elif len(reasons_bgcFa_is_not_calculable) == 0 and len(reasons_bfaGcfa_is_not_calculable) == 0:
            assert bgcFa   is not None
            assert bfaGcfa is not None
            if not eq(bgcFa, bfaGcfa):
                counterexamples.add((aWrapped,bWrapped,cWrapped, f"f(g({b},{c}), {a}) = f({bGc}, {a}) = {bgcFa} ≠ {bfaGcfa} = g({bFa}, {cFa}) = g(f({b},{a}),f({c},{a}))"))
        else:
            pass
    if returnCounterexamples:
        return counterexamples
    return len(counterexamples) == 0


def preserves_partial_order(po_stack, op, returnCounterexamples=False):
    '''
    Given a stack of pfvs and a (unary) function to/from pfvs, this checks 
    whether the image of this function preserves the partial ordering relation 
    of the stack, i.e. whether
      a ≤ b ⇒ f(a) ≤ f(b)
    holds with respect to the input stack elements ≡ whether f is *monotone*.

    By default, this returns a boolean. If returnCounterexamples=True, then this
    returns a (possibly empty) set of counterexamples found.

    NB if the function is partial, then any a,b where at least one of
     - f(a)
     - f(b)
    does not exist is considered a counterexample to the function preserving
    order.
    '''
    # po_stack = hashableArrays_to_stack(po_set)
    po_set = stack_to_set(po_stack)
    allPairs = {(a,b) for a in po_set for b in po_set}
    counterexamples = set()
    for aWrapped, bWrapped in allPairs:
        a,   b       = aWrapped.unwrap(), bWrapped.unwrap()
        aLTEb_before = lte_specification(a, b)
        fa, fb       = op(a), op(b)
        if (fa is None) or (fb is None):
            counterexamples.add(((aWrapped, 
                                  bWrapped, 
                                  aLTEb_before), 
                                 (HashableArray(fa) if fa is not None else None, 
                                  HashableArray(fb) if fb is not None else None, 
                                  None)))
        else:
            aLTEb_after = lte_specification(fa, fb)
            if aLTEb_before and not aLTEb_after:
                counterexamples.add(((aWrapped        , bWrapped         , aLTEb_before), 
                                     (HashableArray(fa), HashableArray(fb), aLTEb_after)))
    if returnCounterexamples:
        return counterexamples
    return len(counterexamples) == 0


def preserves_meet(M, op, returnCounterexamples=False):
    '''
    Given a stack of pfvs and a (unary) function to/from pfvs, this checks 
    whether the function preserves meets, i.e. whether
      f(a) ∧ f(b) = f(a ∧ b)
    holds with respect to the input stack elements.

    By default, this returns a boolean. If returnCounterexamples=True, then this
    returns a (possibly empty) set of counterexamples found.

    NB if the function is partial, then any a,b such that any of
     - f(a)
     - f(b)
     - f(a ∧ b)
    do not exist is considered a counterexample to meet being preserved.
    '''
    # M = hashableArrays_to_stack(Ms)
    Ms = stack_to_set(M)
    allPairs = {(a,b) for a in Ms for b in Ms}
    counterexamples = set()
    for aWrapped, bWrapped in allPairs:
        reasons    = []
        a,   b     = aWrapped.unwrap(), bWrapped.unwrap()
        m_before   = meet_specification( a,  b)
        f_m_before = op(m_before)
        fa, fb     = op(a), op(b)
        m_after    = meet_specification(fa, fb) if (fa is not None) and (fb is not None) else None

        reasons_faMb_is_not_calculable = []
        if f_m_before is None:
            reasons_faMb_is_not_calculable.append(f"f({a} ∧ {b}) dne")
        
        reasons_faMfb_is_not_calculable = []
        if fa is None:
            reasons_faMfb_is_not_calculable.append(f"f({a}) dne")
        if fb is None:
            reasons_faMfb_is_not_calculable.append(f"f({b}) dne")

        if len(reasons_faMb_is_not_calculable) == 0 and len(reasons_faMfb_is_not_calculable) > 0:
            assert f_m_before is not None
            assert m_after    is     None
            reasons.append(f"f({a} ∧ {b}) exists but f({a}) ∧ f({b}) dne")
            reasons.extend(reasons_faMfb_is_not_calculable)
            counterexamples.add(((aWrapped, bWrapped, HashableArray(m_before), HashableArray(f_m_before)), 
                                 (HashableArray(fa) if fa is not None else None, HashableArray(fb) if fb is not None else None),
                                 tuple(reasons)))
        elif len(reasons_faMb_is_not_calculable) > 0 and len(reasons_faMfb_is_not_calculable) == 0:
            assert f_m_before is     None
            assert m_after    is not None
            reasons.append(f"f({a}) ∧ f({b}) exists but f({a} ∧ {b}) dne")
            reasons.extend(reasons_faMb_is_not_calculable)
            counterexamples.add(((aWrapped, bWrapped, HashableArray(m_before)), 
                                 (HashableArray(fa), HashableArray(fb), HashableArray(m_after)),
                                 tuple(reasons)))
        elif len(reasons_faMb_is_not_calculable) == 0 and len(reasons_faMfb_is_not_calculable) == 0:
            assert f_m_before is not None
            assert m_after    is not None
            if not np.array_equal(f_m_before, m_after):
                counterexamples.add(((aWrapped         , bWrapped         , HashableArray(m_before), HashableArray(f_m_before)), 
                                     (HashableArray(fa), HashableArray(fb), HashableArray(m_after))))
        else:
            pass
    if returnCounterexamples:
        return counterexamples
    return len(counterexamples) == 0


def preserves_join(M, op, returnCounterexamples=False):
    '''
    Given a stack of pfvs and a (unary) function to/from pfvs, this checks 
    whether the function preserves joins, i.e. whether
      f(a) ∨ f(b) = f(a ∨ b)
    holds with respect to the input stack elements.

    By default, this returns a boolean. If returnCounterexamples=True, then this
    returns a (possibly empty) set of counterexamples found.

    NB if the function is partial, then any a,b such that any of
     - f(a)
     - f(b)
     - f(a ∨ b)
    do not exist is considered a counterexample to join being preserved.
    '''
    # M = hashableArrays_to_stack(Ms)
    Ms = stack_to_set(M)
    allPairs = {(a,b) for a in Ms for b in Ms}
    counterexamples = set()
    for aWrapped, bWrapped in allPairs:
        reasons    = []
        a,   b     = aWrapped.unwrap(), bWrapped.unwrap()
        j_before   = join_specification( a,  b)
        f_j_before = op(j_before) if j_before is not None else None
        fa, fb     = op(a), op(b)
        j_after    = join_specification(fa, fb) if (fa is not None) and (fb is not None) else None

        reasons_faJb_is_not_calculable = []
        if j_before is None:
            reasons_faJb_is_not_calculable.append(f"{a} ∨ {b} dne")
        if (j_before is not None) and f_j_before is None:
            reasons_faJb_is_not_calculable.append(f"{a} ∨ {b} exists but f({a} ∨ {b}) dne")
        
        reasons_faJfb_is_not_calculable = []
        if fa is None:
            reasons_faJfb_is_not_calculable.append(f"f({a}) dne")
        if fb is None:
            reasons_faJfb_is_not_calculable.append(f"f({b}) dne")
        if (fa is not None) and (fb is not None) and j_after is None:
            reasons_faJfb_is_not_calculable.append(f"f({a}), f({b}) exist but f({a}) ∨ f({b}) dne")

        if len(reasons_faJb_is_not_calculable) == 0 and len(reasons_faJfb_is_not_calculable) > 0:
            assert f_j_before is not None
            assert j_after    is     None
            reasons.append(f"f({a} ∨ {b}) exists but f({a}) ∨ f({b}) dne")
            reasons.extend(reasons_faJfb_is_not_calculable)
            counterexamples.add(((aWrapped, 
                                  bWrapped, 
                                  HashableArray(j_before) if j_before is not None else None, 
                                  HashableArray(f_j_before) if f_j_before is not None else None), 
                                 (HashableArray(fa) if fa is not None else None, 
                                  HashableArray(fb) if fb is not None else None),
                                 tuple(reasons)))
        elif len(reasons_faJb_is_not_calculable) > 0 and len(reasons_faJfb_is_not_calculable) == 0:
            assert f_j_before is     None
            assert j_after    is not None
            reasons.append(f"f({a}) ∨ f({b}) exists but f({a} ∨ {b}) dne")
            reasons.extend(reasons_faJb_is_not_calculable)
            counterexamples.add(((aWrapped, 
                                  bWrapped, 
                                  HashableArray(j_before) if j_before is not None else None), 
                                 (HashableArray(fa) if fa is not None else None, 
                                  HashableArray(fb) if fb is not None else None, 
                                  HashableArray(j_after) if j_after is not None else None),
                                 tuple(reasons)))
        elif len(reasons_faJb_is_not_calculable) == 0 and len(reasons_faJfb_is_not_calculable) == 0:
            assert f_j_before is not None
            assert j_after    is not None
            if not np.array_equal(f_j_before, j_after):
                counterexamples.add(((aWrapped         , bWrapped         , HashableArray(j_before), HashableArray(f_j_before)), 
                                     (HashableArray(fa), HashableArray(fb), HashableArray(j_after))))
        else:
            pass
    if returnCounterexamples:
        return counterexamples
    return len(counterexamples) == 0


def preserves_min(M, op, returnCounterexamples=False):
    '''
    Given a stack of pfvs and a (unary) function to/from pfvs, this checks 
    whether the function preserves the minimum of the stack (if such a minimum 
    exists), i.e. whether
      f(min(stack)) = min(f[stack])

    By default, this returns a boolean. If returnCounterexamples=True, then this
    returns data (if any) concerning why the function does not preserve the 
    minimum.

    NB if the function is partial, then if at least one of
     - f(min(stack))
     - min(f[stack])
    do not exist, then the minimum is considered to not be preserved.
    '''
    Ms            = stack_to_set(M)
    glb           = min_of(M)
    if glb is None:
        if returnCounterexamples:
            return set()
        return True
    glb_out       = op(glb) if glb is not None else None
    if glb_out is None:
        reasons = {(HashableArray(glb), f"f({glb}) dne")}
        print(reasons)
        if returnCounterexamples:
            return reasons
        return False
    
    Ms_out      = {HashableArray(v_out) 
                   for v_out in [op(v.unwrap()) for v in Ms] 
                   if v_out is not None}
    M_out_stack = hashableArrays_to_stack(Ms_out)
    
    assert glb_out is not None
    glb_M_out = min_of(M_out_stack)
    if glb_M_out is None:
        reasons = {(HashableArray(glb), HashableArray(glb_out), f"glb of image dne")}
        print(reasons)
        if returnCounterexamples:
            return reasons
        return False
    else:
        assert glb       is not None
        assert glb_out   is not None
        assert glb_M_out is not None
        if np.array_equal(glb_out, glb_M_out):
            if returnCounterexamples:
                return set()
            return True
        else:
            if returnCounterexamples:
                return {(HashableArray(glb), HashableArray(glb_out), HashableArray(glb_M_out))}
            return False


def preserves_max(M, op, returnCounterexamples=False):
    '''
    Given a stack of pfvs and a (unary) function to/from pfvs, this checks 
    whether the function preserves the maximum of the stack (if such a maximum 
    exists), i.e. whether
      f(max(stack)) = max(f[stack])

    By default, this returns a boolean. If returnCounterexamples=True, then this
    returns data (if any) concerning why the function does not preserve the 
    maximum.

    NB if the function is partial, then if at least one of
     - f(max(stack))
     - max(f[stack])
    do not exist, then the maximum is considered to not be preserved.
    '''
    Ms            = stack_to_set(M)
    lub           = max_of(M)
    if lub is None:
        if returnCounterexamples:
            return set()
        return True
    lub_out       = op(lub) if lub is not None else None
    if lub_out is None:
        reasons = {(HashableArray(lub), f"f({lub}) dne")}
        print(reasons)
        if returnCounterexamples:
            return reasons
        return False
    
    Ms_out      = {HashableArray(v_out) 
                   for v_out in [op(v.unwrap()) for v in Ms] 
                   if v_out is not None}
    M_out_stack = hashableArrays_to_stack(Ms_out)
    
    assert lub_out is not None
    lub_M_out = max_of(M_out_stack)
    if lub_M_out is None:
        reasons = {(HashableArray(lub), HashableArray(lub_out), f"lub of image dne")}
        print(reasons)
        if returnCounterexamples:
            return reasons
        return False
    else:
        assert lub       is not None
        assert lub_out   is not None
        assert lub_M_out is not None
        if np.array_equal(lub_out, lub_M_out):
            if returnCounterexamples:
                return set()
            return True
        else:
            if returnCounterexamples:
                return {(HashableArray(lub), HashableArray(lub_out), HashableArray(lub_M_out))}
            return False


def is_meet_semilattice_homomorphism(M, op, returnCounterexamples=False):
    '''
    Given a stack of pfvs forming a meet semilattice and a (unary) function 
    to/from pfvs, this checks whether the function
     - preserves meets
     - preserves the maximum of the stack (if such a maximum exists)
    with respect to the stack.

    By default, this returns a boolean. If returnCounterexamples=True, then this
    returns data (if any) concerning why the function is not a meet semilattice
    homomorphism.
    '''
    assert is_meet_semilattice(M), f"Input stack is not a meet semilattice:\n{M}"
    preserves_meet_cxs = preserves_meet(M, op, returnCounterexamples=True)
    preserves_lub_cxs  = preserves_max(M, op, returnCounterexamples=True)
    cxs = preserves_meet_cxs.union(preserves_lub_cxs)
    if returnCounterexamples:
        return cxs
    return len(cxs) == 0


def is_join_semilattice_homomorphism(M, op, returnCounterexamples=False):
    '''
    Given a stack of pfvs forming a join semilattice and a (unary) function 
    to/from pfvs, this checks whether the function
     - preserves joins
     - preserves the minimum of the stack (if such a minimum exists)
    with respect to the stack.

    By default, this returns a boolean. If returnCounterexamples=True, then this
    returns data (if any) concerning why the function is not a join semilattice
    homomorphism.
    '''
    assert is_join_semilattice(M), f"Input stack is not a join semilattice:\n{M}"
    preserves_join_cxs = preserves_join(M, op, returnCounterexamples=True)
    preserves_glb_cxs  = preserves_min(M, op, returnCounterexamples=True)
    cxs = preserves_join_cxs.union(preserves_glb_cxs)
    if returnCounterexamples:
        return cxs
    return len(cxs) == 0


def is_lattice_homomorphism(M, op, returnCounterexamples=False):
    '''
    Given a stack of pfvs forming a lattice and a (unary) function to/from pfvs,
    this checks whether the function
     - is a meet semilattice homomorphism
     - is a join semilattice homomorphism
    with respect to the stack.

    By default, this returns a boolean. If returnCounterexamples=True, then this
    returns data (if any) concerning why the function is not a lattice 
    homomorphism.
    '''
    assert is_lattice(M), f"Input stack is not a lattice:\n{M}"
    msl_hm_cxs = is_meet_semilattice_homomorphism(M, op, returnCounterexamples=True)
    jsl_hm_cxs = is_join_semilattice_homomorphism(M, op, returnCounterexamples=True)
    cxs = msl_hm_cxs.union(jsl_hm_cxs)
    if returnCounterexamples:
        return cxs
    return len(cxs) == 0


def is_total_over(M, op, returnCounterexamples=False):
    '''
    Given a stack of pfvs and a (unary) function to/from pfvs, this checks 
    whether the function is defined on every element of the stack.

    By default, this returns a boolean. If returnCounterexamples=True, then this
    returns data (if any) concerning why the function is not total.
    '''
    Ms = stack_to_set(M)
    counterexamples = set()
    for vWrapped in Ms:
        v = vWrapped.unwrap()
        if op(v) is None:
            counterexamples.add(vWrapped)
    if returnCounterexamples:
        return counterexamples
    return len(counterexamples) == 0


def is_endomorphism(M, op, ignoreUndefined=False, returnCounterexamples=False):
    '''
    Given a stack of pfvs and a (unary) function to/from pfvs, this checks 
    whether the image of every pfv in the input stack is some (not necessarily
    distinct) pfv in the input stack.

    By default, this returns a boolean. If returnCounterexamples=True, then this
    returns data (if any) concerning why the function is not an endomorphism.

    If ignoreUndefined=False, then any function that is partial over the input
    stack will NOT be considered an endomorphism. Otherwise, elements of the 
    input stack where the function is undefined will be ignored when considering
    whether the function is an endomorphism.
    '''
    counterexamples = set()
    if not ignoreUndefined:
        non_total_counterexamples = is_total_over(M, op, returnCounterexamples=True)
        counterexamples.union(non_total_counterexamples)

    Ms = stack_to_set(M)
    for vWrapped in Ms:
        v = vWrapped.unwrap()
        v_out = op(v)
        if v_out is not None:
            v_outWrapped = HashableArray(v_out)
            if v_outWrapped not in Ms:
                counterexamples.add((vWrapped, v_outWrapped))
    if returnCounterexamples:
        return counterexamples
    return len(counterexamples) == 0


def fibers(M, op):
    '''
    Given a stack of pfvs and a unary function to/from pfvs, this calculates
    the set of fibers of the function with respect to the input stack:
      - given f: X -> Y,
        fibers(f)(y) = {x | f(x) = y}
    where fibers is a dictionary with keys (either HashableArrays or Nones) and 
    values that are sets of HashableArrays.
    '''
    Ms = stack_to_set(M)
    fibers = dict()
    for vWrapped in Ms:
        v = vWrapped.unwrap()
        v_out = op(v)
        v_outWrapped = HashableArray(v_out) if v_out is not None else None
        if v_outWrapped in fibers:
            fibers[v_outWrapped].add(vWrapped)
        else:
            fibers[v_outWrapped] = {vWrapped}
    return fibers


def is_bijection(M, op, ignoreUndefined=False, returnCounterexamples=False):
    '''
    Given a stack of pfvs and a (unary) function to/from pfvs, this checks 
    whether the function is a bijection.

    By default, this returns a boolean. If returnCounterexamples=True, then this
    returns data (if any) concerning why the function is not a bijection.

    If ignoreUndefined=False, then any function that is partial over the input
    stack will NOT be considered a bijection. Otherwise, elements of the 
    input stack where the function is undefined will be ignored when considering
    whether the function is a bijection.
    '''
    counterexamples = set()
    if not ignoreUndefined:
        non_total_counterexamples = is_total_over(M, op, returnCounterexamples=True)
        counterexamples.union(non_total_counterexamples)

    Ms = stack_to_set(M)
    myFibers = fibers(M, op)
    if ignoreUndefined and None in myFibers:
        del myFibers[None]
    for v_outWrapped in myFibers:
        # v_out = v_outWrapped.unwrap()
        if len(myFibers[v_outWrapped]) > 0:
            counterexamples.add((v_outWrapped, frozenset(myFibers[v_outWrapped])))
    if returnCounterexamples:
        return counterexamples
    return len(counterexamples) == 0


def image_always_satisfies(M, op, pred, returnCounterexamples=False):
    '''
    Given 
     - a stack of pfvs
     - a (unary) function to/from pfvs
     - a predicate on stacks of pfvs
    this checks whether the image of the input stack under the function 
    satisfies the predicate.

    By default, this returns a boolean. If returnCounterexamples=True, then this
    returns data (if any) concerning why the image of the input stack under the 
    function does not satisfy the predicate.
    '''
    Ms          = stack_to_set(M)
    Ms_out      = {HashableArray(v_out) 
                   for v_out in [op(v.unwrap()) for v in Ms] 
                   if v_out is not None}
    M_out_stack = hashableArrays_to_stack(Ms_out)
    cxs         = pred(M_out_stack, returnCounterexamples=True)
    if returnCounterexamples:
        return cxs
    return len(cxs) == 0


def image_always_satisfies_pred_when_domain_does(M, op, pred, returnCounterexamples=False):
    '''
    Given 
     - a stack of pfvs
     - a (unary) function to/from pfvs
     - a predicate on stacks of pfvs
    this checks whether the image of the input stack under the function 
    satisfies the predicate whenever the input stack itself does.

    By default, this returns a boolean. If returnCounterexamples=True, then this
    returns data (if any) concerning why the image of the input stack under the 
    function does not satisfy the predicate.
    '''
    domain_satisfies_pred = pred(M, returnCounterexamples=False)
    if not domain_satisfies_pred:
        if returnCounterexamples:
            return set()
        return True
    
    Ms          = stack_to_set(M)
    Ms_out      = {HashableArray(v_out) 
                   for v_out in [op(v.unwrap()) for v in Ms] 
                   if v_out is not None}
    M_out_stack = hashableArrays_to_stack(Ms_out)
    cxs         = pred(M_out_stack, returnCounterexamples=True)
    if returnCounterexamples:
        return cxs
    return len(cxs) == 0


def interval(u,v):
    '''
    Given two pfvs u,v such that u ≤ v, this returns the stack of pfvs 
      {x | u ≤ x ≤ v}
    Note that this stack is necessarily a bounded lattice.
    '''
    if not lte_specification(u,v):
        return None
    m             = u.shape[0]
    maxSpecDegree = m
    uSpecDegree, vSpecDegree = specification_degree(u), specification_degree(v)
    uDistFromBottom = uSpecDegree
    vDistFromTop    = maxSpecDegree - vSpecDegree
    if uDistFromBottom >= vDistFromTop:
        uUC     = upper_closure(u)
        capMask = lte_specification_stack_left(uUC, v)
        cap     = uUC[capMask.nonzero()]
    else:
        vLC     = lower_closure(v)
        capMask = lte_specification_stack_right(u, vLC)
        cap     = vLC[capMask.nonzero()]
    return cap


def interval_intersection(intervalABounds, intervalBBounds):
    '''
    Given two specification semilattice intervals, each defined by a tuple 
    (max, min) describing the maximum and minimum of each interval, this 
    returns the interval (if it exists) that is the intersection of the two 
    input intervals. (The returned interval will be specified in the same way
    the input intervals are.)
    '''
    maxA, minA = intervalABounds
    maxB, minB = intervalBBounds
    maxCap = meet_specification(maxA, maxB)
    minCap = join_specification(minA, minB)
    if maxCap is None or minCap is None or not lte_specification(minCap, maxCap):
        return None
    return (minCap, maxCap)


############################################################
# GENERATING THE LOWER CLOSURE OF A PARTIAL FEATURE VECTOR #
############################################################

def get_children(pfv, object_inventory=None):
    '''
    Returns pfvs 'one step' below in the specification semilattice.

    If an object_inventory is provided, then only pfvs with a non-empty
    extension will be returned. Note that:
     - if 〚v〛= ∅, then it is *possible* that some child has a non-empty
       extension.
     - if 〚v〛≠ ∅, then it is *necessary* that *all* children have non-empty
       extensions.
    '''
    #TODO support alternative functionality to avoid generating particular
    # children (besides those with nonempty extension)

    m                 = pfv.shape[0]
    specified_indices = pfv.nonzero()[0]
    k                 = specified_indices.shape[0]

    #TODO this step could be optimized if needed/justified
    #TODO it might be ideal to alter this step to never generate children with
    # empty extension (or other properties, cf above)
    despec_masks      = np.ones((k,m), dtype=INT8)
    for i,x in enumerate(specified_indices):
        despec_masks[i,x] = 0
    children          = pfv * despec_masks


    if object_inventory is None:
        return children

    child_extensions                 = extensions(children, object_inventory)
    extensions_nonempty_indicator    = child_extensions.sum(axis=1)
    nonempty_child_indices           = extensions_nonempty_indicator.nonzero()[0]
    children_with_nonempty_extension = children[nonempty_child_indices]
    return children_with_nonempty_extension


def get_parents(pfv, object_inventory=None):
    '''
    Returns pfvs 'one step' above in the specification semilattice.

    If an object_inventory is provided, then only pfvs with a non-empty
    extension will be returned. Note that:
     - if 〚v〛= ∅, then it is *necessary* that *all* parents have empty
       extensions.
     - if 〚v〛≠ ∅, then it is *possible* that some child has a non-empty
       extension.
    '''
    #TODO support alternative functionality to avoid generating particular
    # parents (besides those with nonempty extension)

    m                    = pfv.shape[0]
    specified_mask       = np.abs(pfv, dtype=np.int8)
    nonspecified_mask    = np.logical_not(specified_mask)
    nonspecified_indices = nonspecified_mask.nonzero()[0]
    k                    = nonspecified_indices.shape[0]

    #TODO this step could be optimized if needed/justified
    #TODO it might be ideal to alter this step to never generate parents with
    # empty extension (or other properties, cf above)
    spec_mod = np.zeros((2*k,m), dtype=INT8)
    for i in np.arange(2*k):
        x = 1 if i % 2 == 0 else -1
        j = nonspecified_indices[i // 2]
        spec_mod[i, j] = x
    parents = pfv + spec_mod

    if object_inventory is None:
        return parents

    # my_extension = extension(pfv, object_inventory)
    # current_pfv_is_empty = my_extension.sum() == 0
#     if current_pfv_is_empty:
#         print('foo')
#         return np.empty_like(pfv)[[]]

    parent_extensions               = extensions(parents, object_inventory)
    extensions_nonempty_indicator   = parent_extensions.sum(axis=1)
    nonempty_parent_indices         = extensions_nonempty_indicator.nonzero()[0]
    parents_with_nonempty_extension = parents[nonempty_parent_indices]
    return parents_with_nonempty_extension


# Useful for testing.
# Illustrates general idea behind lower_closure.
def gen_lc(x):
    '''Generates a random element u of ↓x.

    If k = the number of specified indices in x, then, the generative procedure
    for creating an element u of  ↓x is as follows:
      1. A number n≥0 of indices to unspecify is chosen uniformly from [1,k].
      2. n indices are chosen randomly without replacement from among the
      specified ones.
    '''
    specified_indices        = x.nonzero()[0]
    k                        = len(specified_indices)
    num_indices_to_unspecify = random.choice(np.arange(k+1))
    indices_to_unspecify     = np.random.choice(specified_indices,
                                                size=num_indices_to_unspecify,
                                                replace=False)
    u                        = composable_put(x, indices_to_unspecify, 0)
    return u


# Slightly adapted from https://stackoverflow.com/a/42202157
def combinations_np(n, k):
    '''A NumPy-based analogue of itertools.combinations. Assume you have an
    ndarray x with n elements and want to generate all ways of selecting k
    indices of x.

    Let r = (n choose k). This function generates an ndarray of shape (r,n)
    where each column is a set of k indices.
    '''
    a    = np.ones((k, n-k+1), dtype=int)
    a[0] = np.arange(n-k+1)
    for j in range(1, k):
        reps           = (n-k+j) - a[j-1]
        a              = np.repeat(a, reps, axis=1)
        ind            = np.add.accumulate(reps)
        a[j, ind[:-1]] = 1-reps[1:]
        a[j, 0]        = j
        a[j]           = np.add.accumulate(a[j])
    return a


def n_choose_at_most_k_indices(n, k, asMask=True):
    '''An extension of combinations_np. Where combinations_np constructs all
    the ways of choosing a subset of *exactly* k elements from a vector x with
    n elements, this constructs all ways of choosing ≤ k elements, where each
    combination is given by a *row* of the output rather than a column.

    If asMask is True:
    Let r' = (n choose k), and let r = 𝚺_{i=0}^{i=k} (n choose i). This
    generates an ndarray of shape (r,n) where each row is a binary selection
    mask indicating the subset of elements of x included in a particular
    combination of chosen indices.

    If asMask is False:
    This generates a tuple of k+1 ndarrays, where each ndarray directly
    indicates selected indices instead of a binary selection mask.
    '''
    if not asMask:
        exact_results_indices = [np.empty((1,0), dtype=np.int64)] + [combinations_np(n,i).T
                                                                     for i in np.arange(1, k+1)]
        return tuple(exact_results_indices)
    mask = np.concatenate([np.zeros((1,n), dtype=INT8)] +
                          [composable_put_along_axis(np.zeros((int(scipy.special.binom(n,i)), n),
                                                              dtype=INT8),
                                                     combinations_np(n,i).T,
                                                     1,
                                                     axis=1,
                                                     copy_arg=False)
                           for i in np.arange(1, k+1)]).astype(INT8)
    return mask


def lower_closure(x, strict=False, prev_pfvs=None):
    '''The lower closure ↓x of a pfv x is the set of (optionally strictly) less
    specified vectors. If X is the set of all partially specified feature
    vectors, then
        ↓x = {y ∈ X | y ≤ x}

    These are returned as a stack of partial feature vectors (one vector per
    row), with no guarantees about the order of such vectors.


    If m is the total number of features that could be specified, then there
    are O(𝚺_i=1^i=m m choose i) elements in this set.

    If k ≤ m is the exact number of features that are despecifiable in pfv x,
    then there are exactly (𝚺_i=1^i=k k choose i) = 2^k elements in this set.

    If prev_pfvs != None, then
     - it must be an N x m matrix (N ≥ 1)
     - no part of ↓x that overlaps with the lower closure of any pfv in
       prev_will be constructed here
         - Actually, for now they're just eagerly filtered out.
    (This is useful for avoiding redundant computation when constructing sets
    of lower closures.)
    '''
    specified_indices   = x.nonzero()[0]
    k                   = len(specified_indices)

    unspecified_indices = (x == 0).nonzero()[0]

    #There is one element in ↓x for each possible combination of specified
    # indices = a combination of indices of x that can be *un*specified.
    combinations_of_indices_to_unspecify = n_choose_at_most_k_indices(k, k,
                                                                      asMask=True)

    #The goal is to efficiently generate ↓x via Hadamard product of x with a
    # stack of vectors representing 'masks' that each cause a different kind
    # of unspecification.

    #Create a mask with the same shape as x and selection indices in the
    # right place.
    offsets        = np.arange(len(unspecified_indices))
    selection_mask = np.insert(combinations_of_indices_to_unspecify,
                               obj = unspecified_indices - offsets,
                               values = 0, #0s go in the indices whose specification won't be changed
                               axis = 1)   #masks are stacked vertically
    del combinations_of_indices_to_unspecify
    del offsets
    del unspecified_indices
    del specified_indices
    del k

    if prev_pfvs is not None and prev_pfvs.shape[0] != 0:
        meets = meet_specification(x, prev_pfvs) #one meet per row of prev_pfvs

        #reduce meets to minimal subset...
        #...by first eliminating meets that are exact matches of other meets...
        unique_meets = np.flip(np.unique(meets, axis=0), axis=0)
        del meets
        #...and then eliminating meets that contain no information not captured
        # by other unique meets
        #...by calculating whether each meet ≤ any meet in meets
        lte_meets_product = lte_specification_dagwood(unique_meets, unique_meets)
        #...then identifying which meets are only ≤ just one meet (viz. themselves)
        mask_meets_that_are_not_lte_another_meet = lte_meets_product.sum(axis=1) == 1
        del lte_meets_product
        #...and only using these meets
        reduced_meets = unique_meets[mask_meets_that_are_not_lte_another_meet]
        del mask_meets_that_are_not_lte_another_meet
        del unique_meets
        #Motivating example:
        #Suppose a = [1,1,1], prev_pfvs = [c,g] where
        #        c = [1,-1,1] and g = [-1,-1,1]
        #
        #Since meet(a,c) = [1,0,1] and meet(a,g) = [0,0,1]
        # and meet(a,g) ≤ meet(a,c), filtering out meet(a,c) from ↓a
        # means meet(a,c) has *already been* filtered out

        #FIXME use meet_diff_masks to filter out selection_mask
#         del prev_pfvs

#         meet_diff_masks = diff_mask(x, reduced_meets)

#         del reduced_meets

    #By negating the selection mask, we get a stack of vectors that each have
    # 0s where we want to erase (unspecify) a value.
    eraser_mask = np.logical_not(selection_mask).astype(INT8)
    del selection_mask

    my_lower_closure = (x * eraser_mask).astype(INT8)
    del eraser_mask

    if strict:
        my_lower_closure = my_lower_closure[1:] #pop first element (==x)

    if prev_pfvs is not None and prev_pfvs.shape[0] != 0:
#         print(f"reduced_meets = {reduced_meets}")
        lte_lc_and_meets = lte_specification_dagwood(my_lower_closure, reduced_meets)
#         print(f"lte_lc_and_meets = {lte_lc_and_meets}")
        mask_lc_elements_not_lte_any_meet = lte_lc_and_meets.sum(axis=1) == 0
#         print(f"mask_lc_elements_not_lte_any_meet.shape = {mask_lc_elements_not_lte_any_meet.shape}")
#         print(f"mask_lc_elements_not_lte_any_meet = {mask_lc_elements_not_lte_any_meet}")
        del lte_lc_and_meets
        my_lower_closure = my_lower_closure[mask_lc_elements_not_lte_any_meet]
        del mask_lc_elements_not_lte_any_meet
        del reduced_meets

    return my_lower_closure


def upper_closure(x, strict=False):
    '''
    The upper closure ↑x of a pfv x is the set of (optionally strictly) more
    specified vectors. If X is the set of all partially specified feature
    vectors, then
        ↑x = {y ∈ X | x ≤ y}

    This function returns that as an ndarray.
    '''
    #TODO #FIXME make this function match lower_closure in options, performance,
    # and return-type.
    unspecified_indices = (x == 0).nonzero()[0]
    m_x = len(unspecified_indices)
    #There are 2^i elements in ↓x for each possible combination of i unspecified
    #indices.
    combinations_of_indices_to_specify = cat(itertools.combinations(unspecified_indices, i)
                                             for i in range(0,m_x+1))
#     specifications = cat(map(np.array, permutations([-1,1], len(combo)))
#                          for combo in combinations_of_indices_to_specify)
    up_x = np.array(list(composable_put(x, tuple(ind), spec)
                         for ind in combinations_of_indices_to_specify
                         for spec in map(np.array,
                                         itertools.product([-1,1], repeat=len(ind)))), 
                    dtype=INT8)
    return up_x


def lower_closure_BFE(x, object_inventory=None, prev_pfvs=None):
    '''
    Calculates the lower closure of x via breadth-first enumeration...
    '''
    x_hashed = hash_ternary_pfv(x)

    visited = collections.defaultdict(lambda: False)
    queue = []
    queue.append(x_hashed)
    visited[x_hashed] = True

    def adjacent_nodes(hashed_pfv):
        unhashed_pfv = decode_hash(hashed_pfv)
        children     = get_children(unhashed_pfv,
                                    object_inventory=object_inventory)
        return children

    def out_of_bounds(unhashed_pfv):
        if prev_pfvs is None:
            return False
        pfv_is_lte_prev_pfvs = lte_specification_stack_right(unhashed_pfv,
                                                             prev_pfvs)
        return np.any(pfv_is_lte_prev_pfvs)

    while queue:
        current_node = queue.pop(0)
        for adj_node in adjacent_nodes(current_node):
            hashed_node = hash_ternary_pfv(adj_node)
            if not visited[hashed_node] and not out_of_bounds(adj_node):
                queue.append(hashed_node)
                visited[hashed_node] = True

    my_visited_nodes = set(visited.keys())
#     my_visited_nodes = sorted(list(set(visited.keys())))
    return my_visited_nodes


def gather_all_pfvs_with_nonempty_extension(object_inventory, method='eager_filter'):
    '''
    Given an n x m matrix of pfvs specifying an object inventory with n objects
    and m features together with a string specifying a calculation method (one
    of 'eager_filter', 'unique_hash', or 'np.unique'), returns the set of unique
    pfvs in the relevant specification semilattice guaranteed to have non-empty
    extension.

    No guarantee is given on the ordering of elements.

    Relative time complexity is: eager_filter << unique_hash <<< np.unique.
    Peak memory usage is very high for both unique hash and np.unique.
    Relative peak memory usage for eager_filter is unknown at present #FIXME.

    unique_hash and np.unique both independently generate the lower closure of
    every pfv in object_inventory and then use different methods to uniquify the
    resulting matrix of pfvs. Peak memory usage (at least) just prior to
    uniquifying is high as a result.

    eager_filter passes information about what other pfvs a lower closure has
    been computed for to each call to lower_closure(x) and then uses that to
    eagerly filter the full ↓x. This could be further optimized by choosing an
    optimal ordering on elements of the object inventory to minimize wasted
    redundancy checking/filtering computation or by avoiding generation of
    redundant elements altogether.
    '''
    assert method in {'eager_filter', 'unique_hash', 'np.unique'}

    if method == 'eager_filter':
        prefixes_O     = prefixes(object_inventory)
        lower_closures = [lower_closure(o, prev_pfvs=prefixes_O[i])
                          for i,o in enumerate(object_inventory)]
        del prefixes_O
        lc_mat         = np.concatenate(lower_closures)
        del lower_closures
        return lc_mat
    else:
        lower_closures = [lower_closure(o) for o in object_inventory]
        lc_mat         = np.concatenate(lower_closures)
        del lower_closures
        if method == 'np.unique':
            return np.unique(lc_mat, return_index=False, axis=0)
        # elif method == 'hashable_array': #takes the same amount of time as np.unique
            # lc_set = set(lmap(HashableArray,
                              # list(lc_mat)))
        else:
            lc_unhashed = decode_hash(np.array(tuple(set(hash_ternary_pfv(lc_mat))), 
                                               dtype=INT8))
            return lc_unhashed


##########################
# CALCULATING EXTENSIONS #
##########################


def objects_to_extension_vector(observed_objects, object_inventory):
    '''Given
        a set of observed objects (a stack of feature vectors)
        a set of potentially observable objects (another stack of vectors)
    where object_inventory is an ndarray of object vectors with n rows, this
    returns a vector x of length n where
        x[i] = 1 iff O[i] ∈ the set of observed objects
        x[i] = 0 otherwise
    '''
    observed_objects_as_extension = extensions(observed_objects,
                                               object_inventory).sum(axis=0,
                                                                     dtype=INT8)
    return observed_objects_as_extension


def extension_vector_to_objects(extension_vector, object_inventory):
    '''Given
        an extension vector (a binary selection mask) of length n
        a stack of potentially observable objects (one object/row)
    this returns the subset of objects given by the extension vector.

    If x is the extension vector and O denotes the object inventory, then the
    returned stack of vectors S is such that:
        S.shape == (l,m)
        O[i] ∈ S iff x[i] == 1
    where
        m is the number of features
        l is the number of nonzero entries of x.
    '''
    return object_inventory[extension_vector.nonzero()[0]]


#Illustrates efficient extension calculation for a collection of object vectors.
def extension(u, object_inventory):
    '''Returns an ndarray x = 〚u〛 representing the subset of object_inventory
    that partial feature vector u describes, where object_inventory is an
    ndarray of object vectors (each row is an object).

    If
        the object inventory is O with |O| objects
    then
        |x| = |O| ⟺ x.shape = (n,) and O.shape = (n,m)
    and
        x[i] = 1 iff O[i] ∈ 〚u〛
        x[i] = 0 otherwise
    '''
    return lte_specification_stack_right(u, object_inventory)
    # return lte_specification_stack_left(object_inventory, u)


def extensions(S, object_inventory):
    '''Like extension, but efficiently calculates the collection of extensions
    for a collection (stack) of partial feature vectors S, where each row of S
    is a partial feature vector.
    '''
    return (np.equal(object_inventory, S[:, None, :]) |
            np.equal(S, 0)[:, None, :]).prod(axis=2, dtype=INT8)


def get_pfvs_whose_extension_contains(observed_objects):
    '''
    Given
        a set of observed objects (a stack of feature vectors)
    this returns
        the set of partial feature vectors (a stack, one vector per row)
    whose extension must contain the set of observed objects.
    '''
    maximally_specified_compatible_pfv = meet_specification(M=observed_objects)
    return lower_closure(maximally_specified_compatible_pfv, strict=False)


def get_pfvs_whose_extension_is_exactly(observed_objects, object_inventory):
    '''
    Given
        a set of observed objects (a stack of feature vectors)
        a set of potentially observable objects (another stack of vectors)
    this returns
        the set of partial feature vectors (a stack, one vector per row)
    whose extension must be exactly the set of observed objects.
    '''
    observed_objects_as_extension = objects_to_extension_vector(observed_objects,
                                                                object_inventory)

    maximally_specified_compatible_pfv = meet_specification(M=observed_objects)

    #in principle, not all of this needs to be generated
    my_lower_closure = lower_closure(maximally_specified_compatible_pfv,
                                     strict=False)
    my_extensions = extensions(my_lower_closure, object_inventory)

    matching_extensions = np.equal(observed_objects_as_extension,
                                   my_extensions).prod(axis=1)
    selection_mask = matching_extensions
    matching_indices = selection_mask.nonzero()[0]
    matching_partial_feature_vectors = my_lower_closure[matching_indices]
    return matching_partial_feature_vectors
