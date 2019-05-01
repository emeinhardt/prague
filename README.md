# bento-bool
A repo for efficiently calculating and storing the extensions of binary feature vectors with respect to a set of observed objects.

`Calculating Partial Feature Vector Extension.ipynb` documents code for efficiently pre-calculating (and storing) the extension of every partial feature vector (Boolean formula) given a list of bitstrings representing an inventory of fully specified feature vectors (observed objects drawn from a binary feature space). Given such a pre-calculated mapping, a number of other calculations become much easier. E.g. 
 - calculating the set of all partial feature vectors consistent with a set of symbols
 - calculating the set of simplest partial feature vectors consistent with a set of symbols
 - calculating which features (or feature combinations) entail which others.
 
As the notebook describes, further time and space complexity optimization is worth doing, and there are several directions for doing so described in the notebook.
