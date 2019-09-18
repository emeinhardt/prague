from itertools import starmap, product, combinations, chain, permutations
from funcy import *
from functools import reduce

from random import choice

from tqdm import tqdm
from joblib import Parallel, delayed, Memory

import numpy as np
import sparse
import torch

from scipy.special import binom, comb

################################################################################
# GLOBAL CONSTANTS
################################################################################

# Debugging, testing flag
CAREFUL = False


################################################################################
# JOBLIB
################################################################################

#wrapper for interfacing with joblib with preferred defaults set
def par(gen_expr, j=-1, backend='multiprocessing', verbose=10, prefer='processes'):
    return Parallel(n_jobs=j, backend=backend, verbose=verbose, prefer=prefer)(gen_expr)

#often useful in combination with joblib
def identity(x):
    return x


################################################################################
## DEFAULT TYPES AND CODE FOR SWITCHING BETWEEN PYTORCH AND NUMPY TYPES
################################################################################

# default dtype for partial feature vectors and extension vectors alike (though extension vectors could easily be np.uint8)
myint = np.int8
myint_t = torch.int8

# = torch.int8
torch.set_default_tensor_type('torch.CharTensor')

GPU_AVAILABLE = torch.cuda.is_available()

if GPU_AVAILABLE:
    myint_tg = torch.cuda.CharTensor
else:
    print('No GPU available...')


# functions for conveniently converting to/from numpy and pytorch tensors (and devices)
def nt(np_ndarray, to_dtype=myint_t, device='cpu'):
    if device == 'cpu':
        return torch.from_numpy(np_ndarray, dtype=to_dtype)
    elif GPU_AVAILABLE:
       return torch.from_numpy(np_ndarray, dtype=to_dtype).cuda()


def tn(t_tensor, to_dtype=myint):
    if t_tensor.device.type == 'cpu':
        return t_tensor.numpy().astype(to_dtype)
    else:
        return t_tensor.cpu().numpy().


def get_backend(v):
    '''
    Given a numpy ndarray or a torch tensor,
    returns either the module np or the module torch.
    '''
    if type(v) == np.ndarray:
        return np
    elif type(v) == torch.Tensor:
        return torch
    else:
        raise Exception(f'v is neither an ndarray nor a pytorch tensor. Type: {type(v)}')


################################################################################
# COMPOSABLE, LESS STATEFUL VERSIONS OF NUMPY FUNCTIONS
################################################################################

def put_along_axis_(arr, indices, values, axis=None, copy_arg=True):
    '''
    A functional version of np.put_along_axis that returns the
    array it modifies. See the documentation of that function for more details.
    '''
    if copy_arg:
        my_arr = arr.copy()
    else:
        my_arr = arr
    np.put_along_axis(my_arr, indices, values, axis=axis)
    return my_arr


def put_(a, ind, v, mode='raise', copy_arg=True):
    '''
    A functional version of np.put that returns the array it operates on.
    See the documentation for that function for more details.
    '''
    if copy_arg:
        my_a = a.copy()
    else:
        my_a = a
    np.put(a=my_a, ind=ind, v=v, mode=mode)
    return my_a


################################################################################
# FUNCTIONS USEFUL FOR TESTING
################################################################################

def make_random_pfv(num_features):
    '''
    Generates a random partial feature vector as a numpy array.
    '''
    m = num_features
    return np.random.randint(3, size=m, dtype=myint) - 1


def zeros_to_minus_ones(u):
    '''
    Given a tensor or numpy array of all 0s and 1s, this replaces 0s with -1s
    and returns the result as a numpy array.
    '''
    assert ((u == 1) | (u == 0)).all()
    return np.array([x if x == 1 else -1 for x in u], dtype=myint)


def make_random_objects(l, num_features, as_ndarray=False):
    '''
    Generates l random object feature vectors (fully specified) with num_features.

    If as_ndarray is False, they will be returned as a tuple of ndarrays;
    If as_ndarray is True, they will be returned as an ndarray where each row is
    an object.
    '''
    l = actual_num_objects
    m = num_features
    objects = tuple(set([tuple(np.random.randint(2, size=m)) for each in range(actual_num_objects)]))
    objects = tuple(map(lambda o: np.array(o, dtype=myint), objects))
    objects = tuple([zeros_to_minus_ones(o) for o in objects])
    if not as_ndarray:
        return objects
    return np.array(objects)


def make_generator_vectors(num_features, as_ndarray=False):
    '''
    Given a number of features m, this generates the corresponding 2m (pseudo) one-hot vectors.

    If as_ndarray is False, they are returned as a tuple; otherwise they are returned as an ndarray
     - a stack of vectors where each row is a generator vector.
    '''
    basis_vectors = [np.zeros(num_features, dtype=myint) for each in range(num_features)]
    basis_vectors_neg = [np.zeros(num_features, dtype=myint) for each in range(num_features)]
    for i,v in enumerate(basis_vectors):
        v[i] = 1
    for i,v in enumerate(basis_vectors_neg):
        v[i] = -1
    generators = tuple(basis_vectors + basis_vectors_neg)
    if as_ndarray:
        return np.array(generators, dtype=myint)
    return generators


def wf_pfv(v):
    '''
    Indicates whether v is a well-formed partially-specified feature vector.
    '''
    # allowedValues = {-1,0,1}
    return ((v == -1) | (v == 0) | (v == 1)).all()
    # return all([x in allowedValues for x in v])


def wf_tfv(v):
    '''
    Indicates whether v is a well-formed totally-specified feature vector.
    '''
    return ((v == -1) | (v == 1)).all()
    # allowedValues = {-1,1}
    # return all([x in allowedValues for x in v])


################################################################################
# Code for calculating UPPER and LOWER CLOSURES  of partial feature vectors
# with respect to the generality lattice on partial feature vectors
################################################################################


def upper_closure_size_for_fsfvs(num_features):
    '''
    Calculates the size of the upper closure of a fully-specified feature vector with respect
    to a generality lattice on partial feature vectors with m features:

     = 𝚺_i=0^i=m choose(m, i)
    '''
    return np.sum(binom(num_features, np.arange(num_features + 1)))


def comb_index(n, k):
    '''
    Returns an ndarray :: ((n choose k), k), where each row is a sequence of k indices.

    from https://stackoverflow.com/a/16008578
    '''
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)),
                        int, count=count*k)
    return index.reshape(-1, k)


def n_choose_k_indices(n, k, i=0):
    '''
    Returns a tuple of ndarrays. The first element of this tuple is an ndarray :: (k, (n choose k)),
    where each column is a sequence of k indices.

    Seemingly fastest way to generate all n choose k ways of choosing k of n indices.

    From https://stackoverflow.com/a/42202157
    '''
    if k == 1:
        a = np.arange(i, i+n)
        return tuple([a[None, j:] for j in range(n)])
    template = nump(n-1, k-1, i+1)
    full = np.r_[np.repeat(np.arange(i, i+n-k+1),
                           [t.shape[1] for t in template])[None, :],
                 np.c_[template]]
    return tuple([full[:, j:] for j in np.r_[0, np.add.accumulate(
        [t.shape[1] for t in template[:-1]])]])


def n_choose_k_indices_alt(n, k):
    '''
    Returns an ndarray :: (k, (n choose k)), where each column is a sequence of k indices.

    Seemingly second fastest way to generate all n choose k ways of choosing k of n indices.

    From https://stackoverflow.com/a/42202157
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


def n_choose_at_most_k_indices_comb(n, k, as_mask=True):
    '''
    Returns some representation of all the ways to choose at most k indices out of n.

    If asMask is True, the result is a binary matrix (i.e. mask) with k columns and
    𝚺_i=0^i=k n choose i rows. The first row indicates choosing 0 indices, then the
    following rows indicate 1 choice, etc.

    If asMask is False, the result is a tuple indicating the same information, where
    the ith element is a (1,i) ndarray with i choices of indices.
    '''
    my_f = nump2
    my_k = k
    extra_step = False #for n choose n
    if my_f == nump:
        my_f = lambda n, i: nump(n,i)[0]
        my_k = k if k < n else k-1
        extra_step = False if k < n else True


    if not as_mask:
        exact_results_indices = [np.empty((1,0), dtype=np.int64)] + [my_f(n,i).T
                                                              for i in np.arange(1, my_k+1)]
        if extra_step:
            exact_results_indices = exact_results_indices + [np.arange(n, dtype=np.int64)]
        return tuple(exact_results_indices)
#         print(exact_results_indices)
    mask = np.concatenate([np.zeros((1,n), dtype=myint)] + 
                          [put_along_axis_(np.zeros((int(binom(n,i)), n), dtype=myint),
                                           my_f(n,i).T,
                                           1,
                                           axis=1,
                                           copy_arg=False)
                           for i in np.arange(1,my_k+1)])
#     extra_step = False
    if extra_step: #for n choose n
#         print(mask.shape)
#         print(np.ones((n,)).shape)
#         mask = np.stack([mask, np.ones((n,), dtype=myint)], axis=1)
        mask = np.concatenate([mask, np.ones((1,n), dtype=myint)])
    return mask


def upper_closure(x, as_ndarray=True):
    '''
    The upper closure ↑x of a pfv x is the set of vectors that are at least as general (i.e. no more specified)
    as x. Note that this ordering does not depend on the exact set of real objects.

    If as_ndarray is True (this is faster, but eagerly evaluated), the result is returned as an ndarrray of
    shape (c, m), where c = O(𝚺_i=0^i=m m choose i) and m is the number of features (given implicitly by x). 
    If as_ndarray is False, the result is returned as a generator of vectors (ndarrays of length m).

    WARNING: There are O(𝚺_i=0^i=m m choose i) elements in this set.
    '''
    specified_indices = x.nonzero()[0]
    m_x = len(specified_indices)

    unspecified_indices = (x == 0).nonzero()[0]
    offsets = np.arange(len(unspecified_indices))

    #There is one element in ↑x for each possible combination of specified indices.

    if as_ndarray:
    #     print(specified_indices, m_x)
        combinations_of_indices_to_unspecify = n_choose_at_most_k_indices_comb(m_x, m_x, True)
        mask = np.insert(combinations_of_indices_to_unspecify,
                         obj = unspecified_indices - offsets,
                         values = 0,
                         axis = 1)
        eraser_mask = np.logical_not(mask).astype(myint)
        return x * eraser_mask.astype(myint)
#         combinations_of_indices_to_unspecify = [np.take(specified_indices,
#                                                         each)
#                                                 for each in n_choose_at_most_k_indices_comb(m_x, m_x, False)]

#         print(np.sum(binom(m_x, i) for i in np.arange(1, m_x)))
#         print(combinations_of_indices_to_unspecify.shape)
#         print(x.shape)
#         up_x = put_along_axis_(x[None, :], 
#                                combinations_of_indices_to_unspecify,
#                                0,
#                                axis=1,
#                                copy_arg=False)
   else:
    #     #5-10x slower
        combinations_of_indices_to_unspecify = cat(combinations(specified_indices, i)
                                                   for i in range(1,m_x))
        up_x = (put_(x, tuple(ind), 0) for ind in combinations_of_indices_to_unspecify)
        return up_x


def lower_closure(x):
    '''
    The lower closure ↓x of a pfv x is the set of no more general and no less specific vectors.
    This function returns that as a generator.

    WARNING: There are O(𝚺_i=0^i=m choose(m,i) * 2^i) elements in this set.
    '''
    unspecified_indices = (x == 0).nonzero()[0]
    m_x = len(unspecified_indices)
    #There are 2^i elements in ↓x for each possible combination of i unspecified indices.
    combinations_of_indices_to_specify = cat(combinations(unspecified_indices, i)
                                             for i in range(0,m_x))
#     specifications = cat(map(np.array, permutations([-1,1], len(combo)))
#                          for combo in combinations_of_indices_to_specify)
    down_x = (put_(x, tuple(ind), spec)
              for ind in combinations_of_indices_to_specify
              for spec in map(np.array,
                              product([-1,1], repeat=len(ind))))
    return down_x


def gen_uc(x):
    '''
    Generates a random element u of ↑x.
    Generative procedure:
      1. A number n of indices to unspecify is chosen uniformly from among specified ones.
      2. n indices are sampled without replacement from among the specified ones.
    '''
    specified_indices = x.nonzero()[0]
    m_x = len(specified_indices)
    num_indices_to_unspecify = choice(np.arange(0,m_x))
#     assert num_indices_to_unspecify > 0
    indices_to_unspecify = np.random.choice(specified_indices, 
                                            size=num_indices_to_unspecify, 
                                            replace=False)
    u = put_(x, indices_to_unspecify, 0)
    return u


def gen_lc(x):
    '''
    Generates a random element l of ↓x.
    Generative procedure:
      1. A number n of indices to specify is chosen uniformly from among unspecified ones.
      2. n indices are sampled without replacement from among the unspecified ones.
    '''
    unspecified_indices = (x == 0).nonzero()[0]
    m_x = len(unspecified_indices)
    num_indices_to_specify = choice(np.arange(0,m_x))
#     assert num_indices_to_specify > 0
    indices_to_specify = np.random.choice(unspecified_indices, 
                                         size=num_indices_to_specify, 
                                         replace=False)
    possible_specifications = lmap(np.array, product([-1,1], repeat=len(indices_to_specify)))
    if len(possible_specifications) == 0:
        print(x, m_x, unspecified_indices, num_indices_to_specify, indices_to_specify)
    spec = choice(possible_specifications)
    l = put_(x, indices_to_specify, spec)
    return l


def gen_agreeing(x):
    '''
    Generates a random psfv vector r that agrees with x.
    '''
    specified_indices = x.nonzero()[0]
    unspecified_indices = (x == 0).nonzero()[0]
    has_uc = len(specified_indices) > 0
    has_lc = len(unspecified_indices) > 0
    if has_uc and has_lc:
        sample_function = choice([gen_uc, gen_lc])
        return sample_function(x)
    elif has_uc:
        return gen_uc(x)
    elif has_lc:
        return gen_lc(x)
    else:
        raise Exception(f'x has neither an upper nor a lower closure:\n\tx = {x}')


################################################################################
# Agreement
################################################################################

def ag_feature(x,y):
    '''
    Formula:
    (x == 0 or y == 0) or ((x != 0 and y != 0) and (x == y)), where T = 1 and F = 0

    Pattern:
    x = x ⟶ 1
    0 = _ ⟶ 1
    _ = 0 ⟶ 1
    _ = _ ⟶ 0
    '''
    if x == y:
        return True
    elif x == 0:
        return True
    elif y == 0:
        return True
    else:
        return False


def ag_(x,y):
#     return (not x*y) == -1 # <- BAD
    return not (x*y == -1)

def agree(u,v):
    '''
    Given two vectors u and v, returns a binary vector indicating,
    elementwise, whether u and v 'agree'.

    agree(u[i], v[i]) iff (u[i] == 0 or v[i] == 0) or (u[i] == v[i])
    '''
#     return np.array([True if (u[i] == 0 or v[i] == 0) or (u[i] == v[i]) else False
#                      for i in range(len(u))])
    return np.array([1 if (u[i] == 0 or v[i] == 0) or (u[i] == v[i]) else 0
                     for i in range(len(u))], dtype=myint)


def agree_(u,v):
    '''
    Given two vectors u and v, return 1 iff u and v agree at all indices
    and 0 otherwise.
    '''
    ag = agree(u,v)
    return int(ag.all())


def agree_v(u,v):
    '''
    Given two vectors u and v, return 1 iff u and v agree at all indices
    and 0 otherwise.
    '''
    ag = agree(u,v)
    return (~(u*v == -1)).all()


def agree_mat(A,B):
    '''
    Given two matrices A::(n,m) and B::(n,m),
    return C::(n,1) where
    C[i] = 1 iff A[i] and B[i] agree at all indices
    and 0 otherwise.
    '''
    # (x == 0 or y == 0) or ((x != 0 and y != 0) and (x == y))
    A_unspecified = A == 0
    B_unspecified = B == 0
    A_or_B_unspecified = A_unspecified | B_unspecified

    A_specified = A != 0
    B_specified = B != 0
    A_and_B_specified = A_specified & B_specified
    A_equal_B = np.equal(A,B)
    A_B_both_specified_and_equal = A_and_B_specified & A_equal_B

    ag = A_or_B_unspecified | A_B_both_specified_and_equal
#     return ag
    result = np.prod(ag, axis=-1, dtype=myint)
    return result


def agree_m(A, B, axis=0):
    '''
    Given two matrices A::(n,m) and B::(n,m),
    return C::(n,1) where
    C[i] = 1 iff A[i] and B[i] agree at all indices
    and 0 otherwise.

    Fastest such numpy calculation.
    '''
    return (~np.equal(A*B, -1)).prod(axis=axis)


def agree_mat_t(A,B):
    '''
    Given two matrices (torch tensors) A::(n,m) and B::(n,m),
    return C::(n,1) where
    C[i] = 1 iff A[i] and B[i] agree at all indices
    and 0 otherwise.
    '''
    # (x == 0 or y == 0) or ((x != 0 and y != 0) and (x == y))
    A_unspecified = A == 0
    B_unspecified = B == 0
    A_or_B_unspecified = A_unspecified | B_unspecified

    A_specified = A != 0
    B_specified = B != 0
    A_and_B_specified = A_specified & B_specified
    A_equal_B = torch.eq(A,B)
    A_B_both_specified_and_equal = A_and_B_specified & A_equal_B

    ag = A_or_B_unspecified | A_B_both_specified_and_equal
#     return ag
#     result = np.prod(ag, axis=-1, dtype=myint)
    result = torch.zeros([A.shape[0]], dtype=my_dtype, device=A.device)
    result = torch.prod(ag, dim=1,dtype=my_dtype, out=result)
#     result = ag.type(torch.cuda.ByteTensor).all()
    if result.device.type == 'cuda':
        torch.cuda.empty_cache()
    return result#.type(my_torch_type)


def agree_mt(A, B, dim=0):
    '''
    Given two matrices A::(n,m) and B::(n,m),
    return C::(n,1) where
    C[i] = 1 iff A[i] and B[i] agree at all indices
    and 0 otherwise.

    Fastest such pytorch calculation function.
    '''
    return (~torch.eq(A*B, -1 * torch.ones(A.shape, dtype=A.dtype, device=A.device))).prod(dim=dim)


agreement_t = agree_mt
agreement = agree_m


def make_agreeing_vector_pair(pred=None):
    '''
    Generates a random pair of agreeing vectors. If pred is not None, then
    the pair of vectors must also satisfy pred.

    Note that this function scales *poorly* with the number of features m.

    Given that each feature's value is sampled iid and uniformly,
    the probability that two randomly generated features *disagree*
    is 2/9 = p('+-' ∨ '-+'), so the probability of *agreement* is 7/9.
    Therefore the probability of two random feature vectors with m features
    agreeing on all features is (7/9)^m
    '''
    u = make_random_pfv()
    v = make_random_pfv()
    if pred is None:
        while not agree_(u,v):
            u = make_random_pfv()
            v = make_random_pfv()
        return u,v
    while not agree_(u,v) and not pred(u,v):
        u = make_random_pfv()
        v = make_random_pfv()
    return u,v


################################################################################
# Comparison of partial feature vectors by generality/specificity
################################################################################

def compare_generality_rank(u, v):
    '''
    Given two pfvs u, v, returns 
        1 iff u > v
        0 iff u == v
        -1 iff u < v
    where
        u > v
    iff u is more general (= has fewer specified features)
    than v.
    
    (Works on ndarrays and torch tensors.)
    '''
    if type(u) == np.ndarray and type(v) == np.ndarray:
        backend = np
    elif type(u) == torch.Tensor and type(v) == torch.Tensor:
        backend = torch
    else:
        raise Exception('u,v must both either be of type np.ndarray or torch.Tensor')
    
    u_key, v_key = backend.sum(backend.abs(u)), backend.sum(backend.abs(v))
    if u_key == v_key:
        return 0
    else:
        if u_key < v_key:
            return 1
        else:
            return -1

def incomparable_features(a,b):
    '''
    Feature values a and b are incomparable iff
    a is -1 and b is +1 or vice versa.
    '''
    return (a == -1 & b == 1) | (a == 1 & b == -1)


def incomparable(u,v):
    '''
    Indicates elementwise whether u,v are incomparable.

    Agnostic between numpy and pytorch representations.
    '''
    u_backend = get_backend(u)
    if u_backend == get_backend(v):
        backend = u_backend
    else:
        raise Exception('u,v must have the same backend')
    if backend is np:
        return (np.equal(u, -1) & np.equal(v, 1)) | (np.equal(u, 1) & np.equal(v, -1))
    else:
        return (torch.eq(u, -1) & torch.eq(v, 1)) | (torch.eq(u, 1) & torch.eq(v, -1))


def comp_spec_feature(a, b):
    '''
    At the level of a single feature value f
         f ⊆ f, ∀f ∈ {-1,0,+1}
        +1 ⊂ 0
        -1 ⊂ 0
        (-1 and +1 are incomparable)
    
    This function returns 
         0    if a ⊆ b and b ⊆ a
        +1    if a ⊂ b
        -1    if b ⊂ a
   (np.)NaN   if a and b are incomparable
    '''
    if a == b:
        return 0
    elif a == 0:
        return 1
    elif b == 0:
        return -1
    else:
        return np.NaN

comp_spec_feature_vec = np.vectorize(comp_spec_feature)


def compare_spec(u,v):
    '''
    At the level of a single feature value f
         f ⊆ f, ∀f ∈ {-1,0,+1}
        +1 ⊂ 0
        -1 ⊂ 0
        (-1 and +1 are incomparable)
    
    This function returns 
         0    if u ⊆ v and v ⊆ u
        +1    if u ⊂ v
        -1    if v ⊂ u
        NaN   if u and v are incomparable
    *elementwise*

    FIXME: Currently only works for numpy ndarrays.
    '''
    incomparability = incomparable(u,v)
    if incomparability.any():
        incomparable_indices = incomparability.nonzero()[0]
        first_pass = comp_spec_feature_vec(put_(u, 
                                                incomparable_indices,
                                                -99),
                                           put_(v, 
                                                incomparable_indices,
                                                -99)).astype(np.float16)
        second_pass = put_(first_pass, incomparable_indices, np.NaN)
        return second_pass
#         raise Exception('u and v must be *completely* comparable')
    return comp_spec_feature_vec(u,v)
 

def compare_specification(u, v):
    '''
    Given two pfvs u, v, returns 
        +1   iff ⟦v⟧ ⊂ ⟦u⟧
         0   iff ⟦u⟧ == ⟦v⟧
        -1   iff ⟦u⟧ ⊂ ⟦v⟧
        NaN  iff ⟦u⟧ and ⟦v⟧ are incomparable
    where
        ⟦⸱⟧ 
    is defined with respect to the set of *all 
    logically possible* objects given the feature system.
    
    I.e. at the level of a single feature value f
         f ⊆ f, ∀f ∈ {-1,0,+1}
        +1 ⊂ 0
        -1 ⊂ 0
        (-1, +1 are incomparable)
    FIXME: Currently only works for numpy ndarrays.
    '''
#     if type(u) == np.ndarray and type(v) == np.ndarray:
#         backend = np
#     elif type(u) == torch.Tensor and type(v) == torch.Tensor:
#         backend = torch
#     else:
#         raise Exception('u,v must both either be of type np.ndarray or torch.Tensor')
    if incomparable(u,v).any():
        return np.NaN
    
    elementwise_comp = compare_spec(u,v)
    if (elementwise_comp == 0).all():
        return 0
    elif ((elementwise_comp == 0) | (elementwise_comp == 1)).all():
        return 1
    elif ((elementwise_comp == 0) | (elementwise_comp == -1)).all():
        return -1
    else:
        return np.NaN


################################################################################
# Union
################################################################################

 



################################################################################
# Intersection
################################################################################




################################################################################
# Intersection
################################################################################





################################################################################
# Extensions
################################################################################





################################################################################
# Comparison of Extensions
################################################################################






################################################################################
# Map each extension to compatible PFVs
################################################################################






################################################################################
# 
################################################################################
