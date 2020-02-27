# Todo

Development-oriented todo list.

## Tests

 - Add more unit tests for `feature_vector.py` functions (in general).
 - In particular, add unit tests as multiple implementations of core functions in `feature_vector.py` are added.
 
## Project structure

 - `requirements` file.
 - add license.

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
