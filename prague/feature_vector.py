'''
Contains functions for creating and manipulating NumPy ndarrays that represent
(partial) binary feature vectors
'''

import numpy as np
import scipy.special
import scipy.spatial.distance
import random

from funcy import cat

import itertools

from hashlib import sha1

INT8 = np.int8


#################################
# HASHING BALANCED TERNARY PFVS #
#################################


UNSIGNED_PRECISIONS = np.array([8,16,32,64]).astype(np.uint64)
UNSIGNED_MAX_INTS = (2 ** unsigned_precisions) - 1
UNSIGNED_MAX_INTS
MAX_M_FOR_HASHING = 40 # 3^40 < 2^64, but 3^41 > 2^64....

# 64 * np.log(2) / np.log(39) is slightly more than 40
# 23 * np.log(3) / np.log(20) is slightly more than 36 => we'll need np.uint64 precision...


def ternary_pfv_to_trits(pfv):
    '''
    Converts a balanced ternary pfv to base-3.
    '''
    return pfv + 1


def trits_to_ternary_pfv(trits):
    '''
    Converts a base-3 ndarry to balanced ternary.
    '''
    return trits - 1


def trits_to_digits(trits):
    '''
    Converts a base-3 array to base-10.

    (Also converts dtype to np.uint64.)
    '''
    m = trits.shape[0]
    exponents = np.flip(np.arange(m)).astype(np.uint64)
    my_base = 3
    powers = np.power(np.array([my_base], dtype=np.uint64),
                      exponents)
    terms = np.multiply(powers, trits.astype(np.uint64), dtype=np.uint64)
    return terms


def digits_to_trits(digits):
    '''
    Converts a base-10 array to base-3.
    '''
    m = digits.shape[0]
    terms = []
    base10_int = np.sum(digits.astype(np.uint64))
    quotient = base10_int
    while quotient > 3:
        terms.append(quotient % 3)
        quotient = quotient // 3
        if quotient <= 3:
            terms.append(quotient)
    terms = list(reversed(terms))
    return np.array(terms, dtype=INT8)


def digits_to_int(digits, precision=None):
    '''
    Converts a base-10 array to an unsigned int.

    precision defaults to np.uint64.
    '''
    m = digits.shape[0]
    my_type = np.uint64 if precision is None else precision
    base10_int = np.sum(digits.astype(np.uint64), dtype=my_type)
    return base10_int


def hash_ternary_pfv(pfv, precision=None):
    '''
    Converts a balanced ternary pfv to a unique unsigned int corresponding
    to a base-10 representation.

    precision defaults to np.uint64.
    '''
    trits = ternary_pfv_to_trits(pfv)
    digits = trits_to_digits(trits)
    del trits
    my_int = digits_to_int(digits, precision=precision)
    del digits
    return my_int


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
        self.__tight = tight
        self.__wrapped = np.array(arr) if tight else arr
        self.__hash = int(sha1(arr.view(np.uint8)).hexdigest(), 16)


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


def from_feature_dict(d, feature_seq):
    '''
    Given a feature dictionary and an ordering on features, returns a ternary
    vector version of the dictionary.
    '''
    value_map = {'0':0, '-':-1, '+':1}
    value_mapping = lambda v: value_map[v]
    return np.array([value_mapping(d[f]) for f in feature_seq], dtype=INT8)


def to_feature_dict(feature_seq, u, value_map=None):
    '''
    Given a sequence of features, creates an equivalent feature dictionary.
    Assumes feature_seq is ordered appropriately relative to u.

    If value_map is None, the default (0->'0', 1->'+', -1->'-') will be used.
    '''
    n_features = len(feature_seq)
    n_vals = u.shape[0]
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
    no_brackets = s[1:-1]
    split = [each for each in no_brackets.split(' ') if each != '']
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


##################
# PFV DIFFERENCE #
##################


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


def spe_update(a, b):
    '''
    Coerces a to reflect what's specified in b (per an SPE-style unconditioned
    rule analogous to "a ⟶ b"; does not support alpha notation).
    '''
    b_specification_mask = np.abs(b, dtype=INT8)
    b_unspecification_mask = np.logical_not(b_specification_mask)
    prepped_a = a * b_unspecification_mask
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

    m = pfv.shape[0]
    specified_indices = pfv.nonzero()[0]
    k = specified_indices.shape[0]

    #TODO this step could be optimized if needed/justified
    #TODO it might be ideal to alter this step to never generate children with
    # empty extension (or other properties, cf above)
    despec_masks = np.ones((k,m), dtype=INT8)
    for i,x in enumerate(specified_indices):
        despec_masks[i,x] = 0
    children = pfv * despec_masks


    if object_inventory is None:
        return children

    child_extensions = extensions(children, object_inventory)
    extensions_nonempty_indicator = child_extensions.sum(axis=1)
    nonempty_child_indices = extensions_nonempty_indicator.nonzero()[0]
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

    m = pfv.shape[0]
    specified_mask = np.abs(pfv, dtype=np.int8)
    nonspecified_mask = np.logical_not(specified_mask)
    nonspecified_indices = nonspecified_mask.nonzero()[0]
    k = nonspecified_indices.shape[0]

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

    parent_extensions = extensions(parents, object_inventory)
    extensions_nonempty_indicator = parent_extensions.sum(axis=1)
    nonempty_parent_indices = extensions_nonempty_indicator.nonzero()[0]
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
    specified_indices = x.nonzero()[0]
    k = len(specified_indices)
    num_indices_to_unspecify = random.choice(np.arange(k+1))
    indices_to_unspecify = np.random.choice(specified_indices,
                                            size=num_indices_to_unspecify,
                                            replace=False)
    u = composable_put(x, indices_to_unspecify, 0)
    return u


# Slightly adapted from https://stackoverflow.com/a/42202157
def combinations_np(n, k):
    '''A NumPy-based analogue of itertools.combinations. Assume you have an
    ndarray x with n elements and want to generate all ways of selecting k
    indices of x.

    Let r = (n choose k). This function generates an ndarray of shape (r,n)
    where each column is a set of k indices.
    '''
    a = np.ones((k, n-k+1), dtype=int)
    a[0] = np.arange(n-k+1)
    for j in range(1, k):
        reps = (n-k+j) - a[j-1]
        a = np.repeat(a, reps, axis=1)
        ind = np.add.accumulate(reps)
        a[j, ind[:-1]] = 1-reps[1:]
        a[j, 0] = j
        a[j] = np.add.accumulate(a[j])
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


def lower_closure(x, strict=False):
    '''The lower closure ↓x of a pfv x is the set of (optionally strictly) less
    specified vectors. If X is the set of all partially specified feature
    vectors, then
        ↓x = {y ∈ X | y ≤ x}

    These are returned as a stack of partial feature vectors (one vector per
    row), with no guarantees about the order of such vectors.


    If m is the total number of features that could be specified, then there
    are O(𝚺_i=1^i=m m choose i) elements in this set.

    If k ≤ m is the exact number of features that are specified in pfv x, then
    there are exactly (𝚺_i=1^i=k k choose i) = 2^k elements in this set.
    '''
    specified_indices = x.nonzero()[0]
    k = len(specified_indices)

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
    offsets = np.arange(len(unspecified_indices))
    selection_mask = np.insert(combinations_of_indices_to_unspecify,
                               obj = unspecified_indices - offsets,
                               values = 0, #0s go in the indices whose specification won't be changed
                               axis = 1)   #masks are stacked vertically

    #By negating the selection mask, we get a stack of vectors that each have
    # 0s where we want to erase (unspecify) a value.
    eraser_mask = np.logical_not(selection_mask).astype(INT8)

    my_lower_closure = (x * eraser_mask).astype(INT8)
    if strict:
        my_lower_closure = my_lower_closure[1:] #pop first element (==x)
    return my_lower_closure


def upper_closure(x, strict=False):
    '''
    The upper closure ↑x of a pfv x is the set of (optionally strictly) more
    specified vectors. If X is the set of all partially specified feature
    vectors, then
        ↑x = {y ∈ X | x ≤ y}

    This function returns that set as a generator.
    '''
    #TODO #FIXME make this function match lower_closure in options, performance,
    # and return-type.
    unspecified_indices = (x == 0).nonzero()[0]
    m_x = len(unspecified_indices)
    #There are 2^i elements in ↓x for each possible combination of i unspecified
    #indices.
    combinations_of_indices_to_specify = cat(itertools.combinations(unspecified_indices, i)
                                             for i in range(0,m_x))
#     specifications = cat(map(np.array, permutations([-1,1], len(combo)))
#                          for combo in combinations_of_indices_to_specify)
    up_x = (composable_put(x, tuple(ind), spec)
              for ind in combinations_of_indices_to_specify
              for spec in map(np.array,
                              itertools.product([-1,1], repeat=len(ind))))
    return up_x


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
