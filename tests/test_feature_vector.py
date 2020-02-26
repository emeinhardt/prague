#!/usr/bin/env ipython

import pytest
import numpy as np
import prague.feature_vector as fv


a = np.array([-1,  0,  1,  0], dtype=np.int8)
b = np.array([-1,  1, -1,  0], dtype=np.int8)


lte = fv.lte_specification
lte_stack_left = fv.lte_specification_stack_left
lte_stack_right = fv.lte_specification_stack_right


def test_lte_pair():
    assert lte(a,a)
    assert lte(b,b)
    assert not lte(a,b)
    assert not lte(b,a)


lc_a = np.array([[-1,0,1,0],
                 [0,0,1,0],
                 [-1,0,0,0],
                 [0,0,0,0]],dtype=np.int8)

uc_a = np.array([[-1,0,1,0],
                 [-1,1,1,0],
                 [-1,-1,1,0],
                 [-1,0,1,1],
                 [-1,0,1,-1],
                 [-1,1,1,1],
                 [-1,1,1,-1],
                 [-1,-1,1,1],
                 [-1,-1,1,-1]],dtype=np.int8)


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
    assert np.array_equal(meet(a,b), np.array([-1, 0, 0, 0], dtype=np.int8))

def test_meet_ab_vs_M():
    assert np.array_equal(meet(a,b), meet(M=np.array([a,b])))

def test_meet_lca():
    assert np.array_equal(meet(M=lc_a), np.array([0,0,0,0], dtype=np.int8))

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

SFPs = [S0p, S0n,
        S1p,
        S2p, S2n,
        S3p, S3n]

ext = fv.extension
exts = fv.extensions

def test_SFPs():
    assert np.array_equal(ext(np.array([ 1, 0, 0, 0]), O), S0p)
    assert np.array_equal(ext(np.array([-1, 0, 0, 0]), O), S0n)
    assert np.array_equal(ext(np.array([ 0, 1, 0, 0]), O), S1p)
