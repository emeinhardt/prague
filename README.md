# bento-bool
A repo for efficiently calculating and storing the extensions of binary feature vectors with respect to a set of observed objects.

`Generating extensions` documents code for efficiently generating extensions of partial feature vectors with non-empty extensions and mapping each non-empty extension *x* of some partial feature vector with the set of such vectors that either have exactly that extension *x*, or whose extension is a superset of *x*.

`Convert feature matrix tsv to ternary vector` is a notebook for converting a `.tsv` file representing a feature matrix into the format needed by the `Generating extensions` notebook.

TODO
----
1. Migrate code from `Generating extensions` (and add docstrings!) into a module, `roman.py`.
2. Compare code in `Generating extensions`/what ends up in a preliminary `roman.py` with Jack's code (camelcase-titled notebooks) and use the better/better documented version/take the best docstrings and tests from each.
3. There are further optimizations that could be done with the code in `Generating extensions` (possibly already done in Jack's code) that could make online calculation feasible.
4. All code so far takes the entire feature matrix as given and works with that, but typical use cases will only use a fraction of the total object inventory, so efficient support for taking projections would be nice. 

Old code
-----------

`Calculating Partial Feature Vector Extension.ipynb` documents some code for efficiently pre-calculating (and storing) the extension of every partial feature vector (Boolean formula) given a list of bitstrings representing an inventory of fully specified feature vectors (observed objects drawn from a binary feature space). Given such a pre-calculated mapping, a number of other calculations become much easier. E.g. 
 - calculating the set of all partial feature vectors consistent with a set of objects
 - calculating the set of simplest partial feature vectors consistent with a set of objects
 - calculating which features (or feature combinations) entail which others.
 
As the initial main notebook (`Calculating Partial Feature Vector Extension`) describes, further time and space complexity optimization is worth doing, and there are several directions for doing so described in the notebook. Other notebooks are calculational scratchpads for different representations and implementations of key operations on partial feature vectors.
