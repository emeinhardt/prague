# prague
A package for efficiently calculating and storing the extensions of partial feature vectors (a restricted class of Boolean concepts used e.g. in phonology) with respect to a set of observed objects (e.g. speech sounds), and supporting efficient calculation of which partial feature vectors are compatible with a set of observed objects.


## Motivation / context
The motivating context for the code in this repository is supporting scientific research in computational models of sound patterns in human languages ('phonology').

An important feature of human sound patterns is that they usually involve or apply to a *class of similar sounds or sound sequences* - e.g. 'all consonants', 'all consonants at the ends of words', 'all s-like consonants', 'all nasal consonants that occur before vowels', etc. A traditional and useful (if coarse) measure of similarity between sounds has been captured using binary feature vectors, where each feature has some independently-motivated articulatory and/or acoustic interpretation. Every speech sound type, then, corresponds to some completely specified feature vector - but note that the reverse is not, in general, the case. We can pick out a *set* of similar sounds through a *partially* specified feature vector. (For unknown reasons, most human sound patterns apply to a set of sounds that are all similar in a way that can be captured using a *single* partial feature vector: disjunctions of feature descriptions are not commonly necessary.)

Whether the goal is creating tools that aid phonologists in finding a good description of a sound pattern, comparing feature systems or phonological theories or analyses that are dependent on a choice of feature system, or approximately modeling human phonological learning, more basic but critical computational problems include practical and *exact* calculation of things like 

  1. **Enumeration Problem**: Identifying the set of *all* partial feature vectors that pick out at least one non-empty set of sounds.
  2. **Compatibility Problem**: The set of all partial feature vectors that pick out a set of sounds that *includes* some set of sounds *S*.
  3. **Exact Match Problem**: The set of *all* partial feature vectors that pick out exactly some set of sounds *S*.

Because of the combinatorics involved, there's usually *many* partial feature vectors compatible with a particular observed set of speech sounds participating in some sound pattern, and it's usually not immediately obvious what those partial vectors are. (Zooming out from the specific context of phonology, closely related formal topics are *Boolean concept learning* and *version space learning*.) See `algebra_notes.pdf` for a more precise formal description of the problems above (and some scratch notes used for ongoing development).

The main algorithmic challenge here is figuring out how to make it as practical as possible to calculate items 1-3 above for any given feature system. This is not currently a solved problem, nor, to my knowledge something that anyone has attempted to seriously tackle - phonologists either have to do this by hand (using their implicit domain knowledge), settle for heuristics, or only expect to have to find the/some of the *minimally* specified feature vectors with a particular extension (set of picked out/described sounds). The goal here is to see how these problems are solvable exactly and with computational resources plausibly available to a researcher working with machine-readable phonological data, and to then produce Python modules that researchers can use more or less as-is.

Currently, the key advantages offered by `prague` (over e.g. a homerolled solution) include:
 - clear formalization of what the three problems above *are* and what their substructure is.
 - embedding partial feature vectors and the three problems above into extremely well-optimized linear algebra representations and operations that can take advantage of the embarassingly parallel character of most relevant computations with a relatively small memory footprint.


## Status / organization
The repository is currently practically usable as is (particularly if you have lots of RAM), though there is plenty of room for improvement and additional features.

1. The `data` directory contains key tab-seprated value files representing feature systems and speech sound inventories.
2. The main source directory (`prague`) contains code for 
 - converting tab-separated value files specifying a feature system and object (e.g. speech sound) inventory into a list of dictionaries and ultimately into the ternary NumPy ndarray representations used by everything else.
 - manipulating these ternary vectors and solving problems 1-3 above with practical efficiency for most feature systems and inventories a phonologist is likely to want, provided they have access to a server with say, 20-30 GB of RAM. There is definitely room for improvement in both time and space complexity; this is an ongoing area of development with several promising directions.
3. The `demo` directory contains two Jupyter notebooks outlining use of the two main modules (`prague.convert`, `prague.feature_vector`).
4. The `scratch` folder contains a variety of exploratory/proof-of-concept notebooks for finding representations and algorithms that are practically efficient for the feature systems used in phonology. Good algorithms and representations have been found (especially if the user has access to a GPU): relevant calculation for the three problems described above currently takes no more than tens of minutes (rather than hours or days) and only needs to be done *once* for a particular feature system. 


## Roadmap of planned features

**Performance**

1. There's a lot of room for improving the space efficiency of current functions for solving the Enumeration Problem. This is the highest priority right now. The benchmark for feasibility is the enumeration problem on the full `hayes.tsv` feature set and inventory (about 30 and 285 after deduplication, respectively): if this is reasonabily feasible, then few other more practical problems will present any problem at all. Unfortunately, while there's room for algorithmic improvement, peak memory usage is currently >=90GB and time >>10 minutes.
2. Depending on whether the user has lots of RAM, lots of cores, or access to a GPU, it would be nice to have multiple implementations of core functions for solving the three key problems mentioned above and some way of letting the user choose among them.


**Interface**

1. Support for exporting files - or in-memory functionality - for quickly and easily relating a human-readable representation of a partial feature vector and its already-computed extension.
2. All code so far takes the entire feature matrix as given and works with that, but typical use cases (for phonologists) will only use a fraction of the total object inventory, so support for easily defining and taking relevant projections would be nice.
3. Another useful direction feature might be adding features to support use with other software that phonological researchers use - e.g. [`Phonological Corpus Tools`](https://corpustools.readthedocs.io) or [`PanPhon`](https://github.com/dmort27/panphon).
4. CLI, support for usage via `papermill`.


**Installation and Setup**

1. Slim down the `conda` environment/dependency list.
2. Make `prague` installable via `pip` and `conda`.


**Function Analogues**

1. There are analogues of the three main problems for undirected rewrite rules.


**Analysis/Applications**

On top of the Compatibility Problem and the Exact Match problem, each of the motivations for solving the enumeration problem presents a small domain where some auxiliary functions would be useful to have as part of `prague`/to illustrate its use and functionality. Below are some example domains:

1. *Formal language theory and model selection*: Feature systems permit succinct statement of phonotactic restrictions and phonological processes for analysts (linguists). As a hypothesis about (or at least approximate model of) the generalizations that speakers of a language know about their language (and their expectations that similar sounds behave similarly), feature systems also facilitate learning. `prague` facilitates combinatoric calculations of the relevant sizes of hypothesis spaces for function learning, and comparison of what a particular feature system offers over a naive baseline (or another feature system). Mathematically speaking, it is also the case that phonological processes defined in terms of a feature system are functions on elements of a semilattice (or sequences of such elements) and ultimately define a discrete dynamical system. `prague` could facilitate investigation of what the algebraic/order-theoretic properties of different types or different models of phonological processes are.

2. *Bayesian inference*: Given a probability mass function (pmf) defining a prior distribution over partial feature vectors and some simple assumptions about the likelihood function, the extensions calculated by `prague` let you efficiently calculate the marginal probability of a set of observed objects or the posterior probability of a partial feature vector given a set of observed objects.
3. *Feature predictiveness*: What is the implicative structure of a feature system - in general or for the inventory of a particular language? More preceisely: what does a particular partial assignment of values to features tell you about what the other feature values are more or less likely to be? I.e. given a pmf (prior) *p* over possible object types *O*, what is the *pointwise mutual information* between knowing that a (partially hidden) object (chosen according to *p*) has *k* of its *m* features with particular values and knowing that the rest of its unobserved features have a particular assignment of values?
4. *Comparison of two feature systems*...w.r.t. either of the previous two problems.


## Requirements

Use of this package requires Python 3, plus installation (via your choice of e.g. `pip` or `conda`) of `numpy`, `scipy`, `funcy`, and (for development/testing) `pytest`. Demo notebooks also (non-essentially) make use of `tqdm`.


### Installation / usage

**`conda`**

0. Navigate to a folder of your choice.
1. `git clone https://github.com/emeinhardt/prague.git`
2. Navigate to `prague` (`cd prague` on *nix-like OSs).
3. Create a `conda` environment with the relevant dependencies via `conda env create -f prague_env.yml` (follow prompts), and then `conda activate prague`.
4. As long as the repository root directory is on the path/current directory, you should be able to `import prague` as a Python module --- see e.g. the demo notebooks in `prague/demo/`.
5. If you navigate to the repository root directory (`.../prague/`) and (after setting up an appropriate environment) run `pytest`, you will know if everything has been setup correctly.




## Why `prague`?

[Roman Jakobson](https://www.wikiwand.com/en/Roman_Jakobson) is the linguist most strongly associated with the introduction and use of phonological features. Because `jak` is an already-taken package name, and Jakobson was associated with the [Prague Linguistic Circle](https://www.wikiwand.com/en/Prague_linguistic_circle) for much of his career, `prague` seems like a reasonable choice for a single-word package name associated with distinctive feature calculations.

**Alternative package name possibilities:**
 - `bento-bool`
 - `jak`
 - `jakety-stax`
