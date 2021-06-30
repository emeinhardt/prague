from prague.convert import load_objects
import prague.convert
from prague.feature_vector import make_random_objects, load_object_vectors

from prague.feature_vector import HashableArray, stack_to_set, hashableArrays_to_stack

from prague.feature_vector import from_feature_dict, to_feature_dict, to_spe, from_spe
from prague.feature_vector import make_zero_pfv, make_random_fspfv, make_random_pfv

from prague.feature_vector import ternary_pfv_to_trits, trits_to_ternary_pfv
from prague.feature_vector import trits_to_int, int_to_trits
from prague.feature_vector import hash_ternary_pfv, decode_hash

from prague.feature_vector import cartesian_product, cartesian_product_stack
from prague.feature_vector import combinations_np, n_choose_at_most_k_indices

from prague.feature_vector import dual, normalize
from prague.feature_vector import join_naive, join_specification_possible as join_possible, join_specification as join
from prague.feature_vector import lte_specification_dagwood as lte_dagwood
from prague.feature_vector import lte_specification_stack_left as lte_stack_left
from prague.feature_vector import lte_specification_stack_right as lte_stack_right
from prague.feature_vector import lte_specification as lte
from prague.feature_vector import meet_specification as meet
from prague.feature_vector import specification_degree as spec
from prague.feature_vector import minimally_specified as min_spec
from prague.feature_vector import maximally_specified as max_spec
from prague.feature_vector import min_of, max_of
from prague.feature_vector import upper_closure, lower_closure
from prague.feature_vector import lower_closure_BFE
from prague.feature_vector import get_children, get_parents
from prague.feature_vector import gather_all_pfvs_with_nonempty_extension
from prague.feature_vector import interval, interval_intersection

from prague.feature_vector import is_meet_semilattice, is_join_semilattice, is_lattice
from prague.feature_vector import is_bounded_meet_semilattice, is_bounded_join_semilattice, is_bounded_lattice
from prague.feature_vector import is_lower_closure, is_upper_closure
from prague.feature_vector import is_antichain
from prague.feature_vector import is_meet_semilattice_distributive, is_join_semilattice_distributive
from prague.feature_vector import is_lattice_distributive, is_lattice_modular, is_complemented_lattice
from prague.feature_vector import complement_search, complement_exact

from prague.feature_vector import objects_to_extension_vector, extension_vector_to_objects
from prague.feature_vector import extension, extensions
from prague.feature_vector import get_pfvs_whose_extension_contains
from prague.feature_vector import get_pfvs_whose_extension_is_exactly

from prague.feature_vector import hamming, delta_right, delta_down
from prague.feature_vector import linear_transform, despec, spe_update, priority_union
from prague.feature_vector import left_inv_priority_union as left_inv
from prague.feature_vector import right_inv_priority_union as right_inv
from prague.feature_vector import make_rule
from prague.feature_vector import pseudolinear_inverse_possible, pseudolinear_inverse
from prague.feature_vector import pseudolinear_decomposition
from prague.feature_vector import diff
from prague.feature_vector import fibers, is_endomorphism, is_bijection
from prague.feature_vector import image_always_satisfies, image_always_satisfies_pred_when_domain_does
