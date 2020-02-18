'''
Contains composable versions of two NumPy functions used by a variety of files
here.
'''

import numpy as np

#helper functions
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

