from prague.convert import load_objects
import prague.convert
from prague.feature_vector import make_random_objects, load_object_vectors

from prague.feature_vector import HashableArray

from prague.feature_vector import from_feature_dict, to_feature_dict, to_spe, from_spe

from prague.feature_vector import ternary_pfv_to_trits, trits_to_ternary_pfv
from prague.feature_vector import trits_to_digits, digits_to_trits
from prague.feature_vector import digits_to_int, int_to_digits
from prague.feature_vector import hash_ternary_pfv, decode_hash

from prague.feature_vector import lte_specification_stack_left as lte_stack_left
from prague.feature_vector import lte_specification_stack_right as lte_stack_right
from prague.feature_vector import lte_specification as lte
from prague.feature_vector import meet_specification as meet
from prague.feature_vector import upper_closure, lower_closure
from prague.feature_vector import lower_closure_BFE
from prague.feature_vector import get_children, get_parents

from prague.feature_vector import objects_to_extension_vector, extension_vector_to_objects
from prague.feature_vector import extension, extensions
from prague.feature_vector import get_pfvs_whose_extension_contains
from prague.feature_vector import get_pfvs_whose_extension_is_exactly


from prague.feature_vector import hamming, delta_right, delta_down
from prague.feature_vector import linear_transform, despec, spe_update
