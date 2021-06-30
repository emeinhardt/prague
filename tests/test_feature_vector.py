#!/usr/bin/env ipython

import pytest
import numpy as np
import prague.feature_vector as fv
from funcy import lmap
from functools import reduce

grand_union = lambda sets: reduce(set.union, sets, set())

INT8 = np.int8

#########################################
# BASIC TESTS OF FUNDAMENTAL OPERATIONS #
#########################################

a = np.array([-1, 0,+1, 0], dtype=INT8)
b = np.array([-1,+1,-1, 0], dtype=INT8)


#######
# lte #
#######

lte = fv.lte_specification
lte_stack_left = fv.lte_specification_stack_left
lte_stack_right = fv.lte_specification_stack_right


def test_lte_pair():
    assert lte(a,a)
    assert lte(b,b)
    assert not lte(a,b)
    assert not lte(b,a)


lc_a = np.array([[-1, 0,+1, 0],
                 [ 0, 0,+1, 0],
                 [-1, 0, 0, 0],
                 [ 0, 0, 0, 0]],dtype=INT8)

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


########
# meet #
########

meet = fv.meet_specification

def test_meet_ab():
    assert np.array_equal(meet(a,b), np.array([-1, 0, 0, 0], dtype=INT8))

def test_meet_ab_vs_M():
    assert np.array_equal(meet(a,b), meet(M=np.array([a,b])))

def test_meet_lca():
    assert np.array_equal(meet(M=lc_a), np.array([0,0,0,0], dtype=INT8))

def test_meet_uca():
    assert np.array_equal(meet(M=uc_a), a)

################
# extension(s) #
################

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



################
# trit hashing #
################


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
all3VecsSet = fv.stack_to_set(all3Vecs)
all3VecPairsSet = {(a,b) for a in all3VecsSet for b in all3VecsSet}
all3VecTriplesSet = {(a,b,c) for a in all3VecsSet for b in all3VecsSet for c in all3VecsSet}


#######################################
# stack-oriented join implementations #
#######################################

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

#########################################
# priority union implementation testing #
#########################################


def test_priority_union_stack_left_as_expected():
    counterexamples = set()
    for xWrapped in all3VecsSet:
        x = xWrapped.unwrap()
        stackLeftResult = fv.priority_union(all3Vecs, x)
        nonVectorResult = np.array([fv.priority_union(v, x) for v in all3Vecs], dtype=INT8)
        stackLeftResultSet = fv.stack_to_set(stackLeftResult)
        nonVectorResultSet = fv.stack_to_set(nonVectorResult)
        if stackLeftResultSet != nonVectorResultSet:
            counterexamples.add((xWrapped))
    assert len(counterexamples) == 0, f"Counterexamples:\n{counterexamples}"

def test_priority_union_stack_right_as_expected():
    counterexamples = set()
    for xWrapped in all3VecsSet:
        x = xWrapped.unwrap()
        stackRightResult = fv.priority_union(x, all3Vecs)
        nonVectorResult = np.array([fv.priority_union(x, v) for v in all3Vecs], dtype=INT8)
        stackRightResultSet = fv.stack_to_set(stackRightResult)
        nonVectorResultSet = fv.stack_to_set(nonVectorResult)
        if stackRightResultSet != nonVectorResultSet:
            counterexamples.add((xWrapped))
    assert len(counterexamples) == 0, f"Counterexamples:\n{counterexamples}"


###############################################################
# left/right inverse of priority union implementation testing #
###############################################################

def test_priority_union_left_inverse_actual_is_expected_to_be_possible():
    for i,(a,b,c) in enumerate(all3VecIO):
        assert b in fv.left_inv_priority_union(a=a,c=c),  f"{i}:{a},{b},{c}"

def test_priority_union_right_inverse_actual_is_expected_to_be_possible():
    for i,(a,b,c) in enumerate(all3VecIO):
        assert a in fv.right_inv_priority_union(c=c,b=b), f"{i}:{a},{b},{c}"

def test_priority_union_left_inverse_all_expected_to_be_possible_are():
    Ms = fv.stack_to_set(all3Vecs)
    allPairs = {(a,c) for a in Ms for c in Ms}
    for (aWrapped,cWrapped) in allPairs:
        a, c = aWrapped.unwrap(), cWrapped.unwrap()
        li = fv.left_inv_priority_union(a,c)
        if li is not None:
            for b in li:
                assert np.array_equal(fv.priority_union(a,b), c), f"{a}+{b}≠{c}"

def test_priority_union_right_inverse_all_expected_to_be_possible_are():
    Ms = fv.stack_to_set(all3Vecs)
    allPairs = {(c,b) for c in Ms for b in Ms}
    for (cWrapped,bWrapped) in allPairs:
        c, b = cWrapped.unwrap(), bWrapped.unwrap()
        ri = fv.right_inv_priority_union(c,b)
        if ri is not None:
            for a in ri:
                assert np.array_equal(fv.priority_union(a,b), c), f"{a}+{b}≠{c}"

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



# allLC3sNaive = set(lmap(fv.HashableArray,
#                               [fv.lower_closure(v) for v in all3Vecs]))
allLC3sCompressed = set(lmap(lambda M: frozenset(fv.stack_to_set(M)),
                                   [fv.lower_closure(v) for v in all3Vecs]))
allUC3sCompressed = set(lmap(lambda M: frozenset(fv.stack_to_set(M)),
                                   [fv.upper_closure(v) for v in all3Vecs]))


#####################################
# complement implementation testing #
#####################################

def test_complement_implementations_eq_on_LCs():
    counterexamples = set()
    for lcS in allLC3sCompressed:
        lc = fv.hashableArrays_to_stack(lcS)
        for xWrapped in lcS:
            x      = xWrapped.unwrap()
            cStack = fv.complement_search(x, lc)
            c      = np.expand_dims(fv.complement_exact(x, lc), axis=0)
            if not np.array_equal(cStack, c):
                counterexamples.add((lc,x,cStack,c, f"{lc}, {x} | {cStack} ≠ {c}"))
    assert len(counterexamples) == 0, f"Counterexamples:\n{counterexamples}"

###################
# STRUCTURE TESTS #
###################

# basic properties of THE WHOLE SPECIFICATION SEMILATTICE

def test_all_pfvs_together_form_a_bounded_meet_semilattice():
    assert fv.is_bounded_meet_semilattice(all3Vecs)

def test_all_pfvs_together_do_NOT_form_a_join_semilattice():
    assert not fv.is_join_semilattice(all3Vecs)

def test_all_pfvs_together_do_NOT_form_a_distributive_meet_semilattice():
    is_dist = fv.is_meet_semilattice_distributive(all3Vecs)
    # if not is_dist:
    #     cxsWrapped = fv.is_meet_semilattice_distributive(all3Vecs, True)
    #     cxs        = lmap(lambda abx: lmap(lambda v: v.unwrap(), list(abx)), 
    #                             list(cxsWrapped))
    #     cxsPretty = lmap(lambda abx: f"{abx[2]} ≤ {abx[0]} ∧ {abx[1]} but a',b' dne",
    #                            cxs)
    #     print("\n".join(cxsPretty))
    assert not is_dist

# basic properties of LOWER CLOSURES of the specification semilattice

def test_every_lc_defines_a_bounded_lattice():
    for i,v in enumerate(all3Vecs):
        lc = fv.lower_closure(v)
        assert fv.is_bounded_lattice(lc), f"{i}:{v}\n{lc}"

def test_every_lc_defines_a_distributive_lattice():
    for i,v in enumerate(all3Vecs):
        lc = fv.lower_closure(v)
        assert fv.is_lattice_distributive(lc), f"{i}:{v}\n{lc}"

def test_every_lc_defines_a_modular_lattice():
    for i,v in enumerate(all3Vecs):
        lc = fv.lower_closure(v)
        assert fv.is_lattice_modular(lc), f"{i}:{v}\n{lc}"

def test_every_lc_defines_a_complemented_lattice():
    for i,v in enumerate(all3Vecs):
        lc = fv.lower_closure(v)
        assert fv.is_complemented_lattice(lc), f"{i}:{v}\n{lc}"

def test_intersection_of_every_pair_of_LCs_is_the_LC_of_the_meet():
    for i,(a,b) in enumerate([(a,b) for a in all3Vecs for b in all3Vecs]):
        lcA,  lcB  = fv.lower_closure(a) , fv.lower_closure(b)
        lcAs, lcBs = fv.stack_to_set(lcA), fv.stack_to_set(lcB)
        cap        = lcAs.intersection(lcBs)
        meet       = fv.meet_specification(a,b)
        lcMeet     = fv.lower_closure(meet)
        lcMeets    = fv.stack_to_set(lcMeet)
        assert cap == lcMeets, f"{i}:{a},{b}\n{cap}\n{lcMeets}"

# basic properties of UPPER CLOSURES of the specification semilattice

def test_every_uc_defines_a_bounded_meet_semilattice():
    for i,v in enumerate(all3Vecs):
        uc = fv.upper_closure(v)
        assert fv.is_bounded_meet_semilattice(uc), f"{i}:{v}\n{uc}"

def test_NOT_every_uc_defines_a_join_semilattice():
    does_not_define_a_jsl = []
    for i,v in enumerate(all3Vecs):
        uc = fv.upper_closure(v)
        if not fv.is_join_semilattice(uc):#, f"{i}:{v}\n{uc}"
            does_not_define_a_jsl.append(v)
    assert len(does_not_define_a_jsl) > 0

def test_NOT_every_uc_defines_a_distributive_meet_semilattice():
    does_not_define_a_dmsl = []
    for i,v in enumerate(all3Vecs):
        uc = fv.upper_closure(v)
        if not fv.is_meet_semilattice_distributive(uc):#, f"{i}:{v}\n{uc}"
            does_not_define_a_dmsl.append(v)
    assert len(does_not_define_a_dmsl) > 0

def test_intersection_of_every_pair_of_UCs_is_the_UC_of_the_join():
    for i,(a,b) in enumerate([(a,b) for a in all3Vecs for b in all3Vecs]):
        ucA,  ucB  = fv.upper_closure(a) , fv.upper_closure(b)
        ucAs, ucBs = fv.stack_to_set(ucA), fv.stack_to_set(ucB)
        cap        = ucAs.intersection(ucBs)
        join       = fv.join_specification(a,b)
        if join is not None:
            ucJoin     = fv.upper_closure(join)
            ucJoins    = fv.stack_to_set(ucJoin)
            assert cap == ucJoins, f"{i}:{a},{b}\n{cap}\n{ucJoins}"
        else:
            assert len(cap) == 0, f"{i}:{a},{b}\n{cap}"

# properties of PRIORITY UNION

def test_priority_union_is_associative():
    for xWrapped, yWrapped, zWrapped in all3VecTriplesSet:
        x, y, z = xWrapped.unwrap(), yWrapped.unwrap(), zWrapped.unwrap()
        assert np.array_equal(fv.priority_union(fv.priority_union(x,y), z),
                              fv.priority_union(x, fv.priority_union(y,z)))#, f"({x}+{y})+{z} = {fv.priority_union(x,y)} + {z} = fv.priority_union()"

def test_priority_union_does_NOT_left_distribute_over_meet():
    left_dist_prunion_meet_cxs = fv.left_distribution_bin_bin(fv.priority_union, 
                                                              fv.meet_specification, 
                                                              all3Vecs, 
                                                              returnCounterexamples=True)
    assert len(left_dist_prunion_meet_cxs) > 0

def test_priority_union_right_distributes_over_meet():
    right_dist_prunion_meet_cxs = fv.right_distribution_bin_bin(fv.priority_union, 
                                                                fv.meet_specification, 
                                                                all3Vecs, 
                                                                returnCounterexamples=True)
    assert len(right_dist_prunion_meet_cxs) == 0, f"{right_dist_prunion_meet_cxs}"

def test_priority_union_does_NOT_left_distribute_over_join():
    left_dist_prunion_join_cxs = fv.left_distribution_bin_bin(fv.priority_union, 
                                                              fv.join_specification, 
                                                              all3Vecs, 
                                                              returnCounterexamples=True)
    assert len(left_dist_prunion_join_cxs) > 0

def test_priority_union_does_NOT_right_distribute_over_join():
    right_dist_prunion_join_cxs = fv.right_distribution_bin_bin(fv.priority_union, 
                                                                fv.join_specification, 
                                                                all3Vecs, 
                                                                returnCounterexamples=True)
    assert len(right_dist_prunion_join_cxs) > 0

def test_priority_union_right_preserves_partial_order():
    prunion_pres_po_cxs_rights = []
    for xWrapped in all3VecsSet:
        x = xWrapped.unwrap()
        leftArg = lambda left: fv.priority_union(left, x)
        current_prunion_pres_po_cxs_right = fv.preserves_partial_order(all3Vecs, 
                                                                       leftArg, 
                                                                       returnCounterexamples=True)
        prunion_pres_po_cxs_rights.append(current_prunion_pres_po_cxs_right)
    prunion_pres_po_cxs_right = grand_union(prunion_pres_po_cxs_rights)
    assert len(prunion_pres_po_cxs_right) == 0, f"{prunion_pres_po_cxs_right}"

def test_priority_union_left_does_NOT_preserve_partial_order():
    prunion_pres_po_cxs_lefts = []
    for xWrapped in all3VecsSet:
        x = xWrapped.unwrap()
        rightArg = lambda right: fv.priority_union(x, right)
        current_prunion_pres_po_cxs_left = fv.preserves_partial_order(all3Vecs, 
                                                                      rightArg, 
                                                                      returnCounterexamples=True)
        prunion_pres_po_cxs_lefts.append(current_prunion_pres_po_cxs_left)
    prunion_pres_po_cxs_left = grand_union(prunion_pres_po_cxs_lefts)
    assert len(prunion_pres_po_cxs_left) > 0

# NB preserving meets entails preserving order
def test_priority_union_right_preserves_meets():
    prunion_pres_meet_cxs_rights = []
    for xWrapped in all3VecsSet:
        x = xWrapped.unwrap()
        leftArg = lambda left: fv.priority_union(left, x)
        current_prunion_pres_meet_cxs_right = fv.preserves_meet(all3Vecs, 
                                                                leftArg, 
                                                                returnCounterexamples=True)
        prunion_pres_meet_cxs_rights.append(current_prunion_pres_meet_cxs_right)
    prunion_pres_meet_cxs_right = grand_union(prunion_pres_meet_cxs_rights)
    assert len(prunion_pres_meet_cxs_right) == 0, f"{prunion_pres_meet_cxs_right}"

# NB preserving meets entails preserving order
def test_priority_union_left_does_NOT_preserve_meets():
    prunion_pres_meet_cxs_lefts = []
    for xWrapped in all3VecsSet:
        x = xWrapped.unwrap()
        rightArg = lambda right: fv.priority_union(x, right)
        current_prunion_pres_meet_cxs_left = fv.preserves_meet(all3Vecs, 
                                                               rightArg, 
                                                               returnCounterexamples=True)
        prunion_pres_meet_cxs_lefts.append(current_prunion_pres_meet_cxs_left)
    prunion_pres_meet_cxs_left = grand_union(prunion_pres_meet_cxs_lefts)
    assert len(prunion_pres_meet_cxs_left) > 0#, f"{prunion_pres_meet_cxs_left}"

# NB preserving joins entails preserving order
def test_priority_union_right_does_NOT_preserve_joins():
    prunion_pres_join_cxs_rights = []
    for xWrapped in all3VecsSet:
        x = xWrapped.unwrap()
        leftArg = lambda left: fv.priority_union(left, x)
        current_prunion_pres_join_cxs_right = fv.preserves_join(all3Vecs, 
                                                                leftArg, 
                                                                returnCounterexamples=True)
        prunion_pres_join_cxs_rights.append(current_prunion_pres_join_cxs_right)
    prunion_pres_join_cxs_right = grand_union(prunion_pres_join_cxs_rights)
    assert len(prunion_pres_join_cxs_right) > 0#, f"{prunion_pres_join_cxs_right}"

# NB preserving joins entails preserving order
def test_priority_union_left_does_NOT_preserve_joins():
    prunion_pres_join_cxs_lefts = []
    for xWrapped in all3VecsSet:
        x = xWrapped.unwrap()
        rightArg = lambda right: fv.priority_union(x, right)
        current_prunion_pres_join_cxs_left = fv.preserves_join(all3Vecs, 
                                                               rightArg, 
                                                               returnCounterexamples=True)
        prunion_pres_join_cxs_lefts.append(current_prunion_pres_join_cxs_left)
    prunion_pres_join_cxs_left = grand_union(prunion_pres_join_cxs_lefts)
    assert len(prunion_pres_join_cxs_left) > 0#, f"{prunion_pres_join_cxs_left}"


def test_priority_union_right_is_a_meet_SL_HM_over_UCs():
    prunion_mslhm_UCs_cxs_rights = []
    for xWrapped in all3VecsSet:
        x   = xWrapped.unwrap()
        xUC = fv.upper_closure(x)
        leftArg = lambda left: fv.priority_union(left, x)
        current_prunion_mslhm_UCs_cxs_right = fv.is_meet_semilattice_homomorphism(xUC, 
                                                                                  leftArg, 
                                                                                  returnCounterexamples=True)
        prunion_mslhm_UCs_cxs_rights.append(current_prunion_mslhm_UCs_cxs_right)
    prunion_mslhm_UCs_cxs_right = grand_union(prunion_mslhm_UCs_cxs_rights)
    assert len(prunion_mslhm_UCs_cxs_right) == 0, f"{prunion_mslhm_UCs_cxs_right}"

def test_priority_union_left_is_a_meet_SL_HM_over_UCs():
    prunion_mslhm_UCs_cxs_lefts = []
    for xWrapped in all3VecsSet:
        x   = xWrapped.unwrap()
        xUC = fv.upper_closure(x)
        rightArg = lambda right: fv.priority_union(x, right)
        current_prunion_mslhm_UCs_cxs_left = fv.is_meet_semilattice_homomorphism(xUC, 
                                                                                 rightArg, 
                                                                                 returnCounterexamples=True)
        prunion_mslhm_UCs_cxs_lefts.append(current_prunion_mslhm_UCs_cxs_left)
    prunion_mslhm_UCs_cxs_left = grand_union(prunion_mslhm_UCs_cxs_lefts)
    assert len(prunion_mslhm_UCs_cxs_left) == 0, f"{prunion_mslhm_UCs_cxs_left}"

def test_priority_union_right_is_a_lattice_HM_over_LCs():
    prunion_lhm_LCs_cxs_rights = []
    for xWrapped in all3VecsSet:
        x   = xWrapped.unwrap()
        xLC = fv.lower_closure(x)
        leftArg = lambda left: fv.priority_union(left, x)
        current_prunion_lhm_LCs_cxs_right = fv.is_lattice_homomorphism(xLC, 
                                                                       leftArg, 
                                                                       returnCounterexamples=True)
        prunion_lhm_LCs_cxs_rights.append(current_prunion_lhm_LCs_cxs_right)
    prunion_lhm_LCs_cxs_right = grand_union(prunion_lhm_LCs_cxs_rights)
    assert len(prunion_lhm_LCs_cxs_right) == 0, f"{prunion_lhm_LCs_cxs_right}"

def test_priority_union_left_is_a_lattice_HM_over_LCs():
    prunion_lhm_LCs_cxs_lefts = []
    for xWrapped in all3VecsSet:
        x   = xWrapped.unwrap()
        xLC = fv.lower_closure(x)
        rightArg = lambda right: fv.priority_union(x, right)
        current_prunion_lhm_LCs_cxs_left = fv.is_lattice_homomorphism(xLC, 
                                                                      rightArg, 
                                                                      returnCounterexamples=True)
        prunion_lhm_LCs_cxs_lefts.append(current_prunion_lhm_LCs_cxs_left)
    prunion_lhm_LCs_cxs_left = grand_union(prunion_lhm_LCs_cxs_lefts)
    assert len(prunion_lhm_LCs_cxs_left) == 0, f"{prunion_lhm_LCs_cxs_left}"

# properties of LEFT INVERSE PRIORITY UNION

def test_left_inverse_is_associative():
    counterexamples = set()
    for xWrapped, yWrapped, zWrapped in all3VecTriplesSet:
        x, y, z   = xWrapped.unwrap(), yWrapped.unwrap(), zWrapped.unwrap()
        xy        = fv.left_inv_priority_union(x,y)
        xyWrapped = None if xy is None else fv.stack_to_set(xy)
        if xy is None:
            xy_z  = None
        else:
            xy_zWrapped = set()
            for xyPrimeWrapped in xyWrapped:
                xyPrime = xyPrimeWrapped.unwrap()
                if fv.left_inv_priority_union(xyPrime, z) is not None:
                    xy_zWrapped.union(fv.stack_to_set(fv.left_inv_priority_union(xyPrime, z)))
            xy_z = fv.hashableArrays_to_stack(xy_zWrapped) if len(xy_zWrapped) > 0 else None
            # xy_z  = np.unique(fv.hashableArrays_to_stack(
            #     grand_union([
            #         fv.stack_to_set(fv.left_inv_priority_union(xyPrime, z))
            #         for xyPrime in xyWrapped if fv.left_inv_priority_union(xyPrime, z) is not None])
            # ), axis=0)
        yz        = fv.left_inv_priority_union(y,z)
        yzWrapped = None if yz is None else fv.stack_to_set(yz)
        if yz is None:
            x_yz  = None
        else:
            x_yzWrapped = set()
            for yzPrimeWrapped in yzWrapped:
                yzPrime = yzPrimeWrapped.unwrap()
                if fv.left_inv_priority_union(x, yzPrime) is not None:
                    x_yzWrapped.union(fv.stack_to_set(fv.left_inv_priority_union(x, yzPrime)))
            x_yz = fv.hashableArrays_to_stack(x_yzWrapped) if len(x_yzWrapped) > 0 else None
            # x_yz  = np.unique(fv.hashableArrays_to_stack(
            #     grand_union([
            #         fv.stack_to_set(fv.left_inv_priority_union(x, yzPrime))
            #         for yzPrime in yzWrapped if fv.left_inv_priority_union(x, yzPrime) is not None])
            # ), axis=0)
        # x_yz = None if yz is None else fv.left_inv_priority_union(x, yz)
        if xy_z is None and not x_yz is None:
            counterexamples.add((xWrapped, yWrapped, zWrapped))
        elif xy_z is not None and x_yz is None:
            counterexamples.add((xWrapped, yWrapped, zWrapped))
        elif xy_z is not None and x_yz is not None:
            if not np.array_equal(xy_z, x_yz):
                counterexamples.add((xWrapped, yWrapped, zWrapped))
        else:
            pass
    assert len(counterexamples) == 0, f"{counterexamples}"

def test_not_every_left_inverse_is_a_lower_closure():
    counterexamples = set()
    def left_inverse_is_a_lower_closure(i,a,b,c):
        possibleBs    = fv.left_inv_priority_union(a,c)
        possibleBsSet = fv.stack_to_set(possibleBs)
        return possibleBs is None or possibleBsSet in allLC3sCompressed
        # , f"{i}:{a},{b},{c}\n{possibleBsSet}"
    for i,(a,b,c) in enumerate(all3VecIO):
        if not left_inverse_is_a_lower_closure(i,a,b,c):
            counterexamples.add((i,(fv.HashableArray(a),fv.HashableArray(b),fv.HashableArray(c))))
    assert len(counterexamples) > 0, f"no counterexamples!"


def test_every_defined_left_inverse_yields_a_bounded_lattice():
    counterexamples = set()
    for i,(a,b,c) in enumerate(all3VecIO):
        possibleBs = fv.left_inv_priority_union(a,c)
        if not fv.is_bounded_lattice(possibleBs):
            counterexamples.add((i,(fv.HashableArray(a),fv.HashableArray(b),fv.HashableArray(c))))
    assert len(counterexamples) == 0, f"{counterexamples}"



# properties of RIGHT INVERSE PRIORITY UNION

def test_right_inverse_is_associative():
    counterexamples = set()
    for xWrapped, yWrapped, zWrapped in all3VecTriplesSet:
        x, y, z   = xWrapped.unwrap(), yWrapped.unwrap(), zWrapped.unwrap()
        xy        = fv.right_inv_priority_union(x,y)
        xyWrapped = None if xy is None else fv.stack_to_set(xy)
        if xy is None:
            xy_z  = None
        else:
            xy_zWrapped = set()
            for xyPrimeWrapped in xyWrapped:
                xyPrime = xyPrimeWrapped.unwrap()
                if fv.right_inv_priority_union(xyPrime, z) is not None:
                    xy_zWrapped.union(fv.stack_to_set(fv.right_inv_priority_union(xyPrime, z)))
            xy_z = fv.hashableArrays_to_stack(xy_zWrapped) if len(xy_zWrapped) > 0 else None
            # xy_z  = np.unique(fv.hashableArrays_to_stack(
            #     grand_union([
            #         fv.stack_to_set(fv.right_inv_priority_union(xyPrime, z))
            #         for xyPrime in xyWrapped if fv.right_inv_priority_union(xyPrime, z) is not None])
            # ), axis=0)
        yz        = fv.right_inv_priority_union(y,z)
        yzWrapped = None if yz is None else fv.stack_to_set(yz)
        if yz is None:
            x_yz  = None
        else:
            x_yzWrapped = set()
            for yzPrimeWrapped in yzWrapped:
                yzPrime = yzPrimeWrapped.unwrap()
                if fv.right_inv_priority_union(x, yzPrime) is not None:
                    x_yzWrapped.union(fv.stack_to_set(fv.right_inv_priority_union(x, yzPrime)))
            x_yz = fv.hashableArrays_to_stack(x_yzWrapped) if len(x_yzWrapped) > 0 else None
            # x_yz  = np.unique(fv.hashableArrays_to_stack(
            #     grand_union([
            #         fv.stack_to_set(fv.right_inv_priority_union(x, yzPrime))
            #         for yzPrime in yzWrapped if fv.right_inv_priority_union(x, yzPrime) is not None])
            # ), axis=0)
        # x_yz = None if yz is None else fv.right_inv_priority_union(x, yz)
        if xy_z is None and not x_yz is None:
            counterexamples.add((xWrapped, yWrapped, zWrapped))
        elif xy_z is not None and x_yz is None:
            counterexamples.add((xWrapped, yWrapped, zWrapped))
        elif xy_z is not None and x_yz is not None:
            if not np.array_equal(xy_z, x_yz):
                counterexamples.add((xWrapped, yWrapped, zWrapped))
        else:
            pass
    assert len(counterexamples) == 0, f"{counterexamples}"

def test_every_defined_right_inverse_yields_a_bounded_meet_semilattice():
    counterexamples = set()
    for i,(a,b,c) in enumerate(all3VecIO):
        possibleAs = fv.right_inv_priority_union(c,b)
        if not fv.is_bounded_meet_semilattice(possibleAs):
            counterexamples.add((i,(fv.HashableArray(a),fv.HashableArray(b),fv.HashableArray(c))))
    assert len(counterexamples) == 0, f"Counterexamples:\n{counterexamples}"


# properties of INVERVALS and LEFT INVERSE PRIORITY UNION

all3VecPairsLeftLTERightSet = {(a,b) for a,b in all3VecPairsSet 
                                if fv.lte_specification(a.unwrap(),b.unwrap())}
all3VecIntervals = {frozenset(fv.stack_to_set(fv.interval(a.unwrap(),b.unwrap()))) 
                    for a,b in all3VecPairsLeftLTERightSet}


def test_every_interval_is_a_bounded_lattice():
    counterexamples = set()
    for i,intervalWrapped in enumerate(all3VecIntervals):
        interval = fv.hashableArrays_to_stack(intervalWrapped)
        if not fv.is_bounded_lattice(interval):
            counterexamples.add(intervalWrapped)
    assert len(counterexamples) == 0, f"Counterexamples:\n{counterexamples}"


def test_the_intersection_of_every_pair_of_intervals_is_empty_or_an_interval():
    counterexamples = set()
    for i,intervalLeftWrapped in enumerate(all3VecIntervals):
        # intervalLeft = fv.hashableArrays_to_stack(intervalLeftWrapped)
        for j,intervalRightWrapped in enumerate(all3VecIntervals):
            # intervalRight = fv.hashableArrays_to_stack(intervalRightWrapped)
            capSet   = intervalLeftWrapped.intersection(intervalRightWrapped)
            capStack = fv.hashableArrays_to_stack(capSet)
            if len(capSet) > 0:
                # if not fv.is_bounded_lattice(capStack):
                if not capSet in all3VecIntervals:
                    counterexamples.add(((i,intervalLeftWrapped),(j,intervalRightWrapped)))
    assert len(counterexamples) == 0, f"Counterexamples:\n{counterexamples}"


def test_cap_of_every_interval_pair_is_empty_or_an_interval_with_natural_bounds():
    counterexamples = set()
    for i,intervalLeftWrapped in enumerate(all3VecIntervals):
        intervalLeft     = fv.hashableArrays_to_stack(intervalLeftWrapped)
        maxLeft, minLeft = fv.max_of(intervalLeft), fv.min_of(intervalLeft)
        for j,intervalRightWrapped in enumerate(all3VecIntervals):
            intervalRight      = fv.hashableArrays_to_stack(intervalRightWrapped)
            maxRight, minRight = fv.max_of(intervalRight), fv.min_of(intervalRight)
            
            capSet   = intervalLeftWrapped.intersection(intervalRightWrapped)
            capStack = fv.hashableArrays_to_stack(capSet)
            
            est_cap_max = fv.meet_specification(maxLeft, maxRight)
            est_cap_min = fv.join_specification(minLeft, minRight)
            if len(capSet) > 0:
                est_cap = fv.interval(est_cap_min, est_cap_max)
                est_capSet = fv.stack_to_set(est_cap)
                if not est_capSet == capSet:
                    counterexamples.add(((i,intervalLeftWrapped),(j,intervalRightWrapped), (capSet, est_capSet)))
            else:
                assert est_cap_max is None or est_cap_min is None or not fv.lte_specification(est_cap_min, est_cap_max), f"{est_cap_max}, {est_cap_min}"
    assert len(counterexamples) == 0, f"Counterexamples:\n{counterexamples}"

def test_every_defined_left_inverse_yields_a_bounded_lattice_that_is_an_interval():
    counterexamples = set()
    for i,(a,b,c) in enumerate(all3VecIO):
        possibleBs = fv.left_inv_priority_union(a,c)
        possibleBSet = fv.stack_to_set(possibleBs)
        if not possibleBSet in all3VecIntervals:
            counterexamples.add((i,(fv.HashableArray(a),fv.HashableArray(b),fv.HashableArray(c))))
    assert len(counterexamples) == 0, f"{counterexamples}"

