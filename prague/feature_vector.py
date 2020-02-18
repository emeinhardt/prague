'''
Contains functions for creating and manipulating NumPy ndarrays that represent
(partial) binary feature vectors
'''

import numpy as np
import scipy.special
import random

from funcy import cat

import itertools

# from prague.utility import composable_put, composable_put_along_axis

INT8 = np.int8

###########
# Utility #
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

    Intuitively, u ≤ v holds iff v is *less specified* than u. For example,
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
        (u[i] == v[i]) or (v[i] == 0)
    i.e.
      +1 ≤ +1
      -1 ≤ -1
       0 ≤  0
      +1 ≤  0
      -1 ≤  0
    in this semilattice ordering, and and this ordering on feature values is
    extended to vectors in the natural way: this function returns
        1 iff u[i] ≤ v[i] for all i ∈ [0, m-1]
    and
        0 otherwise.
    '''
    return ((u == v) | (v == 0)).all()


def lte_specification_stack_right(u, M, axis=1):
    '''Given a partial feature vector u and a matrix (stack) of partial feature
    vectors M (one vector per row), this efficiently calculates whether
        u ≤ M[i]
    for each of the i vectors. In other words, this checks membership of each
    M[i] in the upper closure of u.

    If
        u.shape == (m,)
    and
        M.shape == (k,m)
    then
        lte_specification_stack_right(u,M).shape == (k,)
    '''
    return (np.equal(u, M) | np.equal(M, 0)).prod(axis=axis, dtype=INT8)


def lte_specification_stack_left(M, u, axis=1):
    '''Given a partial feature vector u and a matrix (stack) of partial feature
    vectors M (one vector per row), this efficiently calculates whether
        M[i] ≤ u
    for each of the i vectors. In other words, this checks membership of each
    M[i] in the lower closure of u.

    If
        u.shape == (m,)
    and
        M.shape == (k,m)
    then
        lte_specification_stack_left(M,u).shape == (k,)
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
        return np.sign( np.equal.reduce(M, axis=0) * np.sum(M, axis=0), dtype=INT8)
    else:
        raise Exception('Provide exactly two vectors u,v or else a stack M.')


############################################################
# GENERATING THE UPPER CLOSURE OF A PARTIAL FEATURE VECTOR #
############################################################


# Useful for testing.
# Illustrates general idea behind upper_closure.
def gen_uc(x):
    '''Generates a random element u of ↑x.

    If k = the number of specified indices in x, then, the generative procedure
    for creating an element u of  ↑x is as follows:
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


def upper_closure(x, strict=False):
    '''The upper closure ↑x of a pfv x is the set of (optionally strictly) less
    specified vectors. If X is the set of all partially specified feature
    vectors, then
        ↑x = {y ∈ X | x ≤ y}

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

    #There is one element in ↑x for each possible combination of specified
    # indices = a combination of indices of x that can be *un*specified.
    combinations_of_indices_to_unspecify = n_choose_at_most_k_indices(k, k,
                                                                      asMask=True)

    #The goal is to efficiently generate ↑x via Hadamard product of x with a
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

    my_upper_closure = (x * eraser_mask).astype(INT8)
    if strict:
        my_upper_closure = my_upper_closure[1:] #pop first element (==x)
    return my_upper_closure


def lower_closure(x, strict=False):
    '''
    The lower closure ↓x of a pfv x is the set of (optionally strictly) more
    specified vectors. If X is the set of all partially specified feature
    vectors, then
        ↓x = {y ∈ X | y ≤ x}

    This function returns that set as a generator.
    '''
    #TODO #FIXME make this function match upper_closure in options, performance,
    # and return-type.
    unspecified_indices = (x == 0).nonzero()[0]
    m_x = len(unspecified_indices)
    #There are 2^i elements in ↓x for each possible combination of i unspecified
    #indices.
    combinations_of_indices_to_specify = cat(itertools.combinations(unspecified_indices, i)
                                             for i in range(0,m_x))
#     specifications = cat(map(np.array, permutations([-1,1], len(combo)))
#                          for combo in combinations_of_indices_to_specify)
    down_x = (composable_put(x, tuple(ind), spec)
              for ind in combinations_of_indices_to_specify
              for spec in map(np.array,
                              itertools.product([-1,1], repeat=len(ind))))
    return down_x


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
    return lte_specification_stack_left(object_inventory, u)


def extensions(S, object_inventory):
    '''Like extension, but efficiently calculates the collection of extensions
    for a collection (stack) of partial feature vectors S, where each row of S
    is a partial feature vector.
    '''
    return (np.equal(object_inventory, S[:, None, :]) |
            np.equal(S, 0)[:, None, :]).prod(axis=2, dtype=INT8)


def get_pfvs_whose_extension_contains(observed_objects):
    '''Given
        a set of observed objects (a stack of feature vectors)
    this returns
        the set of partial feature vectors (a stack, one vector per row)
    whose extension must contain the set of observed objects.
    '''
    maximally_specified_compatible_pfv = meet_specification(M=observed_objects)
    return upper_closure(maximally_specified_compatible_pfv, strict=False)


def get_pfvs_whose_extension_is_exactly(observed_objects, object_inventory):
    '''Given
        a set of observed objects (a stack of feature vectors)
        a set of potentially observable objects (another stack of vectors)
    this returns
        the set of partial feature vectors (a stack, one vector per row)
    whose extension must be exactly the set of observed objects.
    '''
    observed_objects_as_extension = extensions(observed_objects,
                                               object_inventory)

    maximally_specified_compatible_pfv = meet_specification(M=observed_objects)

    #in principle, not all of this needs to be generated
    my_upper_closure = upper_closure(maximally_specified_compatible_pfv,
                                     strict=False)
    my_extensions = extensions(my_upper_closure, object_inventory)

    matching_extensions = np.equal(observed_objects_as_extension,
                                   my_extensions).prod(axis=1)
    selection_mask = matching_extensions
    matching_indices = selection_mask.nonzero()[0]
    matching_partial_feature_vectors = my_upper_closure[matching_indices]
    return matching_partial_feature_vectors
