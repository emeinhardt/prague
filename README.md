# prague
A repo for efficiently calculating and storing the extensions of partial feature vectors (a restricted class of Boolean concepts used e.g. in phonology) with respect to a set of observed objects (e.g. speech sounds), and supporting efficient calculation of which partial feature vectors are compatible with a set of observed objects.

## Motivation / context
The motivating context for the code in this repository is supporting scientific research in computational models of sound patterns in human languages ('phonology').

An important feature of human sound patterns is that they usually involve or apply to a *class of similar sounds or sound sequences* - e.g. 'all consonants', 'all consonants at the ends of words', 'all s-like consonants', 'all nasal consonants that occur before vowels', etc. A traditional and useful (if coarse) measure of similarity between sounds has been captured using binary feature vectors, where each feature has some independently-motivated articulatory and/or acoustic interpretation. Every speech sound type, then, corresponds to some completely specified feature vector - but note that the reverse is not, in general, the case. We can pick out a *set* of similar sounds through a *partially* specified feature vector. (For unknown reasons, most human sound patterns apply to a set of sounds that are all similar in a way that can be captured using a *single* partial feature vector: disjunctions of feature descriptions are not commonly necessary.)

Whether the goal is creating tools that aid phonologists in finding a good description of a sound pattern, comparing phonological theories, or approximately modeling human phonological learning, more basic but critical computational problems include practical and *exact* calculation of things like 

  1. **Enumeration Problem**: Identifying the set of *all* partial feature vectors that pick out at least one non-empty set of sounds.
  2. **Compatibility Problem**: The set of all partial feature vectors that pick out a set of sounds that *includes* some set of sounds *S*.
  3. **Exact Match Problem**: The set of *all* partial feature vectors that pick out exactly some set of sounds *S*.

Because of the combinatorics involved, there's usually *many* partial feature vectors compatible with a particular observed set of speech sounds participating in some sound pattern, and it's usually not immediately obvious what those partial vectors are. (Zooming out from the specific context of phonology, closely related formal topics are *Boolean concept learning* and *version space learning*.)

The main algorithmic challenge here is figuring out how to make it as practical as possible to calculate items 1-3 above for any given feature system. This is not currently a solved problem, nor, to my knowledge something that anyone has attempted to seriously tackle - phonologists either have to do this by hand (using their implicit domain knowledge), settle for heuristics, or only expect to have to find the/some of the *minimally* specified feature vectors with a particular extension (set of picked out/described sounds). The goal here is to see how these problems are solvable exactly and with computational resources plausibly available to a researcher working with machine-readable phonological data, and to then produce Python modules that researchers can use more or less as-is.


## Status / organization
The repository is currently practically usable as is (particularly if you have lots of RAM), though there is plenty of room for improvement and additional features.

1. The `data` directory contains two key tab-seprated value files representing two feature systems and speech sound inventories.
2. The main source directory (`prague`) contains code for 
 - converting tab-separated value files specifying a feature system and object (e.g. speech sound) inventory into a list of dictionaries and ultimately into the ternary NumPy ndarray representations used by everything else.
 - manipulating these ternary vectors and solving problems 1-3 above with practical efficiency for most feature systems and inventories a phonologist is likely to want, provided they have access to a server with say, 20-30 GB of RAM. There is definitely room for improvement in both time and space complexity; this is an ongoing area of development with several promising directions.
3. The `demo` directory contains two Jupyter notebooks outlining use of the two main modules (`prague.convert`, `prague.feature_vector`).
4. The `scratch` folder contains a variety of exploratory/proof-of-concept notebooks for finding representations and algorithms that are practically efficient for the feature systems used in phonology. Good algorithms and representations have been found (especially if the user has access to a GPU): relevant calculation for the three problems described above currently takes no more than tens of minutes (rather than hours or days) and only needs to be done *once* for a particular feature system. 


## Roadmap of planned features

**Performance**
1. There's a lot of room for improving the space efficiency of current functions for solving the Enumeration Problem. This is the highest priority right now.
2. Depending on whether the user has lots of RAM, lots of cores, or access to a GPU, it would be nice to have multiple implementations of core functions for solving the three key problems mentioned above and some way of letting the user choose among them.


**Interface**
1. Support for exporting files - or in-memory functionality - for quickly and easily relating a human-readable representation of a partial feature vector and its already-computed extension.
2. All code so far takes the entire feature matrix as given and works with that, but typical use cases (for phonologists) will only use a fraction of the total object inventory, so support for easily defining and taking relevant projections would be nice.
3. Another useful direction feature might be adding features to support use with other software that phonological researchers use - e.g. [`Phonological Corpus Tools`](https://corpustools.readthedocs.io) or [`PanPhon`](https://github.com/dmort27/panphon).

## Requirements

Python 3, `numpy`, `scipy`, `funcy`, and (for development/testing) `pytest`. Demo notebooks also (non-essentially) make use of `tqdm`.

## Why `prague`?

[Roman Jakobson](https://www.wikiwand.com/en/Roman_Jakobson) is the linguist most strongly associated with the introduction and use of phonological features. Also `jak` is an already-taken package name.
