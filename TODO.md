# Todo

Development-oriented todo list.

## Documentation and demos

 - Update demo notebook to use functions now integrated into `feature_vector.py`:
   - `symbol_to_feature_vector`
   - `feature_vector_to_symbols`
   - `extension_to_symbols`
   - `symbol_to_feature_dict`
   - `pfv_to_fd`
 - Update demo notebook with whatever rewrite-rule related functionality is there.
 - Replace notes pdf with something more recent (slides?)

## Tests

 - Add more unit tests for `feature_vector.py` functions (in general).
 - In particular, add unit tests as multiple implementations of core functions in `feature_vector.py` are added.
 
## Project structure

 - update/trim dependencies.
 - `requirements` file.
 - add license.
 
## Core features

 - Refactor the following functions for stack-compatibility if possible:
   - `left_inv_priority_union`
   - `right_inv_priority_union`
   - `spe_update` (might already be OK?)
   - `pseudolinear_inverse_possible`
   - `pseudolinear_inverse`
   - `pseudolinear_decomposition`
   - `join_specification_possible`
   - `join_specification`
   - `normalize`
 - Add functionality for solving the function analogues of the three problems described in the `readme`.

## Implementation / performance features

 - Add support for more memory-efficient and memory-safe implementations from existing code.
 - Add support for `joblib` from existing code.
 - Add support for `PySpark` and a library (e.g. `TileDB`) that will efficiently 
   support concurrent reads/writes.
 - Add support for `PyTorch` + gpus from existing code.
 

## Interface features

 - Better support for creating and being able to use an association between
   symbols (or object dictionaries) and associated ternary ndarray feature vectors.
 - Add support for converting between conventional feature vector strings, feature dictionaries, and ternary vectors.
 - Add support for taking and creating projections of a loaded inventory.
 - Add CLI support.
 - Add default support/nice integration with `PanPhon`-exportable data
   structures/formats.
 - Add default support/nice integration with `Phonological Corpus Tools`-exportable data
   structures/formats.
