'''
Module to take
 - a tab-separated value file specifying a binary or ternary feature matrix.
   - the file is assumed to have a header row indicating feature labels
   - each non-header row represents an object's feature vector
   - features are assumed to be {`+`,`-`,`0`} (following phonological
     convention) by default
 - a (potentially -- comma-separated -- list of) column name(s) indicating any
   columns not containing feature-value information (e.g. an object's label =
  IPA symbol in the case of phonological feature matrices).

and write
 - the ordered list of feature names (reflecting the ordering in the input
   `.tsv`) to a `.txt` file
 - a serialized numpy ndarray `.npy` file representing the unique feature
   vectors of the `.tsv` file as a matrix, with each row corresponding to
   an object and feature values represented as `1`, `-1`, or `0`
 - a tab-separated-value file with `+` replaced with `1` and `-` replaced with
   `-1`.

Note that if your feature matrix contains multiple objects that have the same
featural description, this notebook will detect and record as much, but the
final matrix it produces will only have one row for each unique feature vector.
'''

# from os import getcwd, chdir, listdir, path, mkdir, makedirs

from funcy import *

from copy import deepcopy

import numpy as np

import csv


def load_objects(input_filepath, delimiter='\t', quoting=csv.QUOTE_NONE,
               quotechar='@'):
    '''
    Returns a list of objects (dictionaries) found in the .tsv file at
    `input_filepath`.

    Other arguments are passed to csv.DictReader; see that constructor for
    documentation.
    '''
    objects = []

    with open(input_filepath, encoding='utf-8-sig') as csvfile:
        my_reader = csv.DictReader(csvfile, delimiter=delimiter,
                                   quoting=quoting, quotechar=quotechar)
        for row in my_reader:
            #print(row)
            objects.append(row)
    return objects


def have_universal_feature_definitions(objects, behavior='Exception'):
    '''
    Returns True iff all objects are defined for the same set of features.

    If behavior is 'Exception', then this function will raise an exception
    if this property does not hold of the set of objects.
    '''
    features = lmap(lambda d: tuple(d.keys()),
                    objects)
    # values = lmap(lambda d: set(d.values()),
    #               objects)
    result = all(features[0] == features[i] for i in range(len(features)))
    if behavior == 'Exception':
        assert result, f"Not all objects have the same feature set:\nFeatures:{features}"
    return result


def are_identical(a, b, features_to_ignore=None):
    '''
    Compares two objects (dictionaries) for equality of value, ignoring keys in
    `features_to_ignore`.

    Objects are assumed to have the same set of features.
    '''
    if features_to_ignore is None:
        features_to_ignore = set()
    features = a.keys()
    for k in features:
        if k not in features_to_ignore:
            if a[k] != b[k]:
                return False
    return True


def get_matches(obj, objects, features_to_ignore=None):
    '''
    Returns M ⊆ `objects` that match `obj`, ignoring `features_to_ignore` as
    a list
    '''
    return [o for o in objects
       if are_identical(obj, o, features_to_ignore=features_to_ignore)]


def has_duplicates(obj, objects, features_to_ignore=None):
    '''
    Returns True iff `obj` has any duplicates (up to features_to_ignore) in
    `objects`.
    '''
    return len(get_matches(obj, objects, features_to_ignore)) > 1


def objects_are_unique(objects, features_to_ignore=None, behavior='Exception'):
    '''
    Returns True iff all objects are unique (excluding features in
    `features_to_ignore`).

    If behavior is 'Exception', then this function will raise an exception if
    this property does not hold of the set of objects.
    '''
    duplicates_detected = any(has_duplicates(o, objects, features_to_ignore)
                              for o in objects)
    if behavior == 'Exception' and duplicates_detected:
        raise Exception(f"Object set contains duplicates (up to features_to_ignore = {features_to_ignore}):\n{objects}")
    return not duplicates_detected


def sanitized_objects(objects, duplicate_behavior='ignore', features_to_ignore=None):
    '''
    Given a collection of objects (dicts),
     - checks that all objects are defined for the same set of features. If
       they're not, this function will raise an exception.

    If duplicate_behavior is `Exception`, this will also check if there are
    any duplicate objects (with equality up to `features_to_ignore`) and raises
    an exception if so.

    Returns True if no exceptions are raised.
    '''
    have_universal_feature_definitions(objects, behavior='Exception')

    if duplicate_behavior == 'Exception':
        objectsWithDuplicates = [o for o in objects
                                 if has_duplicates(o, objects,
                                                   features_to_ignore)]
        assert len(objectsWithDuplicates) == 0, f'Object set contains duplicates.'
    return True


valueRemap = {'+':1,
              '-':-1,
              '0':0}


def remapValue(v):
    '''
    Remaps '+'/'-'/'0' to 1/-1/0, respectively.
    '''
    return valueRemap[v]


def remapValues(d):
    '''
    Remaps '+'/'-'/'0' to 1/-1/0, respectively, for every key in d.
    Other values are left unchanged.

    (Note: mutates `d` in-place.)
    '''
    for k in d:
        if d[k] in valueRemap:
            d[k] = remapValue(d[k])
    return d


def preprocess_objects(objects, keys_to_remove=None):
    '''
    Given a sanitized collection of objects (dicts) and keys to be removed
    (e.g. symbol columns), this function creates a copy of objects, and then
     - remaps values to integers
     - removes the keys in keys_to_remove

    and returns the resulting collection of feature vectors.
    '''
    remapped_objects = lmap(remapValues, deepcopy(objects))
    if keys_to_remove is not None:
        remapped_objects = lmap(lambda d: omit(d, keys_to_remove),
                                remapped_objects)
    return remapped_objects


def to_ternary_feature_vectors(objects, remove_duplicates=True, feature_ordering=None):
    '''
    Given a preprocessed collection of N objects (dicts) and an optional
    ordering of the M features of the objects, this returns an M x N ternary
    NumPy ndarray representing the collection.

    If feature_ordering is not specified, then the features of the first object
    will be sorted and used.

    If remove_duplicates is False (or if objects contains no duplicates), this
    will preserve the ordering (if any) in objects.

    If remove_duplicates is True, this function will return a tuple where the
    second value is a list of deleted indices.
    '''
    if feature_ordering is None:
        features = tuple(sorted(list(objects)[0].keys()))
    else:
        features = tuple(feature_ordering)

    def to_tuple(o):
        return tuple([o[f] for f in features])

    as_tuples = lmap(to_tuple, objects)
    deleted_indices = []
    if remove_duplicates:
        tuples_seen = set()
        for i,o in enumerate(as_tuples):
            if o in tuples_seen:
                as_tuples[i] = None
                deleted_indices.append(i)
            else:
                tuples_seen.add(o)
        while None in as_tuples:
            as_tuples.remove(None)

    objects_np = np.array(as_tuples, dtype=np.int8)

    if not remove_duplicates:
        return objects_np
    else:
        return objects_np, deleted_indices


def export_ternary_feature_vectors(objects_np, feature_list,
                                 object_output_filepath,
                                 feature_sequence_filepath):
    '''
    Writes the object matrix and the sequence of feature labels to the specified filepaths.

    The object matrix is saved using `np.save` (i.e. as .npy file) and the feature labels
    are written to a textfile.
    '''
    with open(feature_sequence_filepath, 'w') as feature_file:
        for feature_name in feature_list:
            feature_file.write(feature_name + '\n')

    np.save(object_output_filepath, objects_np)



# def process(input_filepath, delimiter='\t',quoting=csv.QUOTE_NONE, quotechar='@',
#           )
