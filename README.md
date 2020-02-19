# prague
A repo for efficiently calculating and storing the extensions of partial feature vectors (a restricted class of Boolean concepts) with respect to a set of observed objects, and supporting efficient calculation related to checking equivalence of partial feature vectors.

## Motivation / context
The motivating context for the code in this repository is supporting scientific research in computational models of sound patterns in human languages ('phonology').

An important feature of human sound patterns is that they usually involve or apply to a *class of similar sounds or sound sequences* - e.g. 'all consonants', 'all consonants at the ends of words', 'all s-like consonants', 'all nasal consonants that occur before vowels', etc. A traditional and useful (if coarse) measure of similarity between sounds has been captured using binary feature vectors, where each feature has some independently-motivated articulatory and/or acoustic interpretation. Every speech sound type, then, corresponds to some completely specified feature vector - but note that the reverse is not, in general, the case. We can pick out a *set* of similar sounds through a *partially* specified feature vector. (For unknown reasons, most human sound patterns apply to a set of sounds that are all similar in a way that can be captured using a *single* partial feature vector: disjunctions of feature descriptions are not commonly necessary.)

Whether the goal is creating tools that aid phonologists in finding a good description of a sound pattern, comparing phonological theories, or approximately modeling human phonological learning, more basic but critical computational problems include practical and *exact* calculation of things like 

  1. Whether any two partial feature vectors *u,v* pick out exactly the same set of speech sounds.
  2. The set of *all* partial feature vectors that pick out exactly some set of sounds *S*.
  3. The set of all partial feature vectors that pick out a set of sounds that *includes* some set of sounds *S*.

Because of the combinatorics involved, there's usually *many* partial feature vectors compatible with a particular observed set of speech sounds participating in some sound pattern, and it's usually not immediately obvious what those partial vectors are. (Zooming out from the specific context of phonology, closely related formal topics are *Boolean concept learning* and *version space learning*.)

The main algorithmic challenge here is figuring out how to make it as practical as possible to calculate items 1-3 above for any given feature system. This is not currently a solved problem, nor, to my knowledge something that anyone has attempted to seriously tackle - phonologists either have to do this by hand (using their implicit domain knowledge) or settle for heuristics. The goal here is to see how these problems are solvable exactly and with computational resources plausibly available to a researcher working with machine-readable phonological data, and to then produce Python modules that researchers can use more or less as-is.


## Roadmap / status
The repository currently contains exploratory/proof-of-concept notebooks for finding representations and algorithms that are practically efficient for the feature systems used in phonology. Good algorithms and representations have been found (assuming the user has access to a GPU): relevant calculation for the three problems described above currently takes no more than tens of minutes (rather than hours or days) and only needs to be done *once* for a particular feature system. 

As proof-of-concept code with no expected user or developer audience in the short term besides myself and Jack, the code currently needs to be cleaned up, documented a bit more, and moved out of notebooks into more polished modules; this is in progress.

Looking further ahead: while the best current code is clearly practical for the setting of phonological feature systems (with e.g. many redundant features, a tiny ratio of actual objects vs. logically describable objects, the use of conjunctive formulas for concepts), it's not clear what the asymptotic complexity of key operations is. This is scientifically interesting question about feature systems as approximate models of the hypothesis spaces humans use in learning sound patterns, and interesting for determining when the algorithms here will generalize to domains other than phonology.

Another useful direction feature might be adding features to support use with other software that phonological researchers use - e.g. [`Phonological Corpus Tools`](https://corpustools.readthedocs.io) or [`PanPhon`](https://github.com/dmort27/panphon).


## Structure of the code
`Generating extensions` documents code for efficiently generating extensions of partial feature vectors with non-empty extensions and mapping each non-empty extension *x* of some partial feature vector with the set of such vectors that either have exactly that extension *x*, or whose extension is a superset of *x*.

`Convert feature matrix tsv to ternary vector` is a notebook for converting a `.tsv` file representing a feature matrix into the format needed by the `Generating extensions` notebook.

TODO
----
1. Migrate code from `Generating extensions` (and add docstrings!) into a module, `roman.py`.
2. Compare code in `Generating extensions`/what ends up in a preliminary `roman.py` with Jack's code (underscore_titled notebooks) and use the better/better documented version/take the best docstrings and tests from each.
3. There are further optimizations that could be done with the code in `Generating extensions` (possibly already done in Jack's code) that could make online calculation feasible.
4. All code so far takes the entire feature matrix as given and works with that, but typical use cases will only use a fraction of the total object inventory, so support for easily defining and taking relevant projections would be nice.

Old code
-----------

`Calculating Partial Feature Vector Extension.ipynb` documents some code for efficiently pre-calculating (and storing) the extension of every partial feature vector (Boolean formula) given a list of bitstrings representing an inventory of fully specified feature vectors (observed objects drawn from a binary feature space). Given such a pre-calculated mapping, a number of other calculations become much easier. E.g. 
 - calculating the set of all partial feature vectors consistent with a set of objects
 - calculating the set of simplest partial feature vectors consistent with a set of objects
 - calculating which features (or feature combinations) entail which others.
 
As the initial main notebook (`Calculating Partial Feature Vector Extension`) describes, further time and space complexity optimization is worth doing, and there are several directions for doing so described in the notebook. Other notebooks are calculational scratchpads for different representations and implementations of key operations on partial feature vectors.
