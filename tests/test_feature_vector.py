#!/usr/bin/env ipython

import pytest
import numpy as np
import prague.feature_vector as fv
import funcy

INT8 = np.int8

a = np.array([-1, 0,+1, 0], dtype=INT8)
b = np.array([-1,+1,-1, 0], dtype=INT8)



lte = fv.lte_specification
lte_stack_left = fv.lte_specification_stack_left
lte_stack_right = fv.lte_specification_stack_right


def test_lte_pair():
    assert lte(a,a)
    assert lte(b,b)
    assert not lte(a,b)
    assert not lte(b,a)


lc_a = np.array([[-1, 0,+1, 0],
                 [0 , 0,+1, 0],
                 [-1, 0, 0, 0],
                 [0 , 0, 0, 0]],dtype=INT8)

uc_a = np.array([[-1, 0,+1, 0],
                 [-1,+1,+1, 0],
                 [-1,-1,+1, 0],
                 [-1, 0,+1,+1],
                 [-1, 0,+1,-1],
                 [-1,+1,+1,+1],
                 [-1,+1,+1,-1],
                 [-1,-1,+1,+1],
                 [-1,-1,+1,-1]],dtype=INT8)


def test_lte_lc():
    for each_v in lc_a:
        assert lte(each_v, a), f"{each_v}, {a}"
        assert np.array_equal(each_v, a) or not lte(a, each_v), f"{each_v}, {a}"


def test_lte_uc():
    for each_v in uc_a:
        assert lte(a, each_v), f"{a}, {each_v}"
        assert np.array_equal(a, each_v) or not lte(each_v, a), f"{a}, {each_v}"


def test_lte_uc_stack():
    assert lte_stack_right(a, uc_a).all(), f"{lte_stack_right(a, uc_a)}"


def test_lte_lc_stack():
    assert lte_stack_left(lc_a, a).all(), f"{lte_stack_left(lc_a, a)}"


meet = fv.meet_specification

def test_meet_ab():
    assert np.array_equal(meet(a,b), np.array([-1, 0, 0, 0], dtype=INT8))

def test_meet_ab_vs_M():
    assert np.array_equal(meet(a,b), meet(M=np.array([a,b])))

def test_meet_lca():
    assert np.array_equal(meet(M=lc_a), np.array([0,0,0,0], dtype=INT8))

def test_meet_uca():
    assert np.array_equal(meet(M=uc_a), a)


# sample object set over 4 features
O = np.array([[-1, 0, 1, 1],
              [ 1, 1,-1, 1],
              [ 1, 1,-1,-1],
              [-1,-1, 1, 1],
              [-1, 0,-1,-1]])
# extensions of some single feature patterns
S0p = np.array([0,1,1,0,0])
S0n = np.array([1,0,0,1,1])
S1p = np.array([0,1,1,0,0])
S2p = np.array([1,0,0,1,0])
S2n = np.array([0,1,1,0,1])
S3p = np.array([1,1,0,1,0])
S3n = np.array([0,0,1,0,1])

SFP_exts = [S0p, S0n,
            S1p,
            S2p, S2n,
            S3p, S3n]
SFP_exts_np = np.array(SFP_exts, dtype=INT8)

my_S = np.array([[ 1, 0, 0, 0],  #S0p
                 [-1, 0, 0, 0],  #S0n
                 [ 0, 1, 0, 0]], #S1p
                dtype=INT8)
my_S_exts = np.array([S0p, S0n, S1p], dtype=INT8)

ext = fv.extension
exts = fv.extensions

def test_SFPs():
    assert np.array_equal(ext(np.array([ 1, 0, 0, 0]), O), S0p)
    assert np.array_equal(ext(np.array([-1, 0, 0, 0]), O), S0n)
    assert np.array_equal(ext(np.array([ 0, 1, 0, 0]), O), S1p)

def test_SFPs_stack():
    assert np.array_equal(exts(my_S, O), my_S_exts)


ternary_pfv_to_trits = fv.ternary_pfv_to_trits
trits_to_int = fv.trits_to_int
int_to_trits = fv.int_to_trits
trits_to_ternary_pfv = fv.trits_to_ternary_pfv

def test_pfv_to_trits():
    assert np.array_equal(ternary_pfv_to_trits(a), np.array([0,1,2,1], dtype=INT8))
    assert np.array_equal(ternary_pfv_to_trits(b), np.array([0,2,0,1], dtype=INT8))
    assert np.array_equal(ternary_pfv_to_trits(np.vstack([a,b])), 
                          np.array([[0,1,2,1],
                                    [0,2,0,1]], 
                                   dtype=INT8))

def test_trits_to_pfv():
    assert np.array_equal(trits_to_ternary_pfv(np.array([0,1,2,1], dtype=INT8)), 
                          a)
    assert np.array_equal(trits_to_ternary_pfv(np.array([0,2,0,1], dtype=INT8)),
                          b)
    assert np.array_equal(trits_to_ternary_pfv(np.array([[0,1,2,1], 
                                                         [0,2,0,1]], 
                                                        dtype=INT8)), 
                          np.vstack([a,b]))

# def test_trits_pfv_inverse():
# #     ternary_pfv_to_trits
# #     trits_to_ternary_pfv
#     pass

def test_trits_to_ints():
    assert trits_to_int(np.array([0,1,2,1], dtype=INT8)) == 16
    assert trits_to_int(np.array([0,2,0,1], dtype=INT8)) == 19
    assert np.array_equal(trits_to_int(np.vstack([[0,1,2,1],
                                                  [0,2,0,1]])), 
                          np.array([16, 19]))

def test_ints_to_trits():
    assert np.array_equal(int_to_trits(16, 4), np.array([0,1,2,1], dtype=INT8))
    assert np.array_equal(int_to_trits(19, 4), np.array([0,2,0,1], dtype=INT8))
    assert np.array_equal(int_to_trits(np.array([16, 19]), 4), 
                          np.vstack([[0,1,2,1], [0,2,0,1]]))

# def test_trits_ints_inverse():
# #     trits_to_ints
# #     ints_to_trits
#     pass

all3Vecs = np.array([[x,y,z] for x in {+1, 0,-1} 
                             for y in {+1, 0,-1} 
                             for z in {+1, 0,-1}], dtype=INT8)


def test_join_specification_possible_stack_implementations_eq():
    Ms = fv.stack_to_set(all3Vecs)
    allPairs = {(a,b) for a in Ms for b in Ms}
    allJoinablePairsA = {(a,b) for (a,b) in allPairs 
                         if fv.join_specification_possible(a.unwrap(), b.unwrap())}
    allJoinablePairsB = {(a,b) for (a,b) in allPairs 
                         if fv.join_specification_possible_stack(np.array([a.unwrap(), b.unwrap()], dtype=INT8))}
    missingFromB = allJoinablePairsA - allJoinablePairsB
    missingFromA = allJoinablePairsB - allJoinablePairsA
    assert allJoinablePairsA == allJoinablePairsB, f"Missing from stack-based result:\n{missingFromB}\nMissing from pairwise result:\n{missingFromA}"

def test_join_specification_stack_implementations_eq():
    Ms = fv.stack_to_set(all3Vecs)
    allPairs = {(a,b) for a in Ms for b in Ms}
    allJoinablePairs = {(a,b) for (a,b) in allPairs 
                        if fv.join_specification_possible(a.unwrap(), b.unwrap())}
    joinsA = {fv.HashableArray(fv.join_specification(a.unwrap(),b.unwrap()))
              for a,b in allJoinablePairs}
    joinsB = {fv.HashableArray(fv.join_specification_stack(np.array([a.unwrap(), b.unwrap()], dtype=INT8)))
              for a,b in allJoinablePairs}
    missingFromB = joinsA - joinsB
    missingFromA = joinsB - joinsA
    assert joinsA == joinsB, f"Missing from stack-based result:\n{missingFromB}\nMissing from pairwise result:\n{missingFromA}"


all3VecIO = [(a,b,fv.priority_union(a,b)) for a in all3Vecs for b in all3Vecs]

def test_priority_union_left_inverse():
    for i,(a,b,c) in enumerate(all3VecIO):
        assert b in fv.left_inv_priority_union(a=a,c=c),  f"{i}:{a},{b},{c}"

def test_priority_union_right_inverse():
    for i,(a,b,c) in enumerate(all3VecIO):
        assert a in fv.right_inv_priority_union(c=c,b=b), f"{i}:{a},{b},{c}"
            

def test_priority_union_left_inverse_implementations_are_eq():
    for i,(a,b,c) in enumerate(all3VecIO):
        oldAnswer = fv.stack_to_set(fv.left_inv_priority_union(a=a,c=c))
        newAnswer = fv.stack_to_set(fv.left_inv_priority_union_old(a=a,c=c))
        missingFromNew = oldAnswer - newAnswer
        extraInNew     = newAnswer - oldAnswer
        assert oldAnswer == newAnswer, f"{i}:{a},{b},{c}\n{missingFromNew}\n{extraInNew}"

def test_priority_union_right_inverse_implementations_are_eq():
    for i,(a,b,c) in enumerate(all3VecIO):
        oldAnswer = fv.stack_to_set(fv.right_inv_priority_union(c=c,b=b))
        newAnswer = fv.stack_to_set(fv.right_inv_priority_union_old(c=c,b=b))
        missingFromNew = oldAnswer - newAnswer
        extraInNew     = newAnswer - oldAnswer
        assert oldAnswer == newAnswer, f"{i}:{a},{b},{c}\n{missingFromNew}\n{extraInNew}"


# allLC3sNaive = set(funcy.lmap(fv.HashableArray,
#                               [fv.lower_closure(v) for v in all3Vecs]))
allLC3sCompressed = set(funcy.lmap(lambda M: frozenset(fv.stack_to_set(M)),
                                   [fv.lower_closure(v) for v in all3Vecs]))

def test_intersection_of_every_pair_of_LCs_is_the_LC_of_the_meet():
    for i,(a,b) in enumerate([(a,b) for a in all3Vecs for b in all3Vecs]):
        lcA,  lcB  = fv.lower_closure(a) , fv.lower_closure(b)
        lcAs, lcBs = fv.stack_to_set(lcA), fv.stack_to_set(lcB)
        cap        = lcAs.intersection(lcBs)
        meet       = fv.meet_specification(a,b)
        lcMeet     = fv.lower_closure(meet)
        lcMeets    = fv.stack_to_set(lcMeet)
        assert cap == lcMeets, f"{i}:{a},{b},{c}\n{cap}\n{lcMeets}"        

def test_not_every_left_inverse_is_a_lower_closure():
    counterexamples = set()
    def left_inverse_is_a_lower_closure(i,a,b,c):
        possibleBs    = fv.left_inv_priority_union(a,c)
        possibleBsSet = fv.stack_to_set(possibleBs)
        return possibleBsSet in allLC3sCompressed
        # , f"{i}:{a},{b},{c}\n{possibleBsSet}"
    for i,(a,b,c) in enumerate(all3VecIO):
        if not left_inverse_is_a_lower_closure(i,a,b,c):
            counterexamples.add((i,(fv.HashableArray(a),fv.HashableArray(b),fv.HashableArray(c))))
    assert len(counterexamples) > 0, f"no counterexamples!"

