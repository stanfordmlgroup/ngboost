"""Serialization utilities for NGBoost models to JSON and Universal Binary JSON formats."""

import base64
import binascii
import json
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.tree import DecisionTreeRegressor

try:
    import ubjson
    UBJSON_AVAILABLE = True
except ImportError:
    UBJSON_AVAILABLE = False


def tree_to_dict(tree: DecisionTreeRegressor) -> Dict[str, Any]:
    """
    Convert a sklearn DecisionTreeRegressor to a JSON-serializable dictionary.
    
    Uses base64-encoded pickle for the tree structure since sklearn's tree
    structure is read-only and cannot be reconstructed from arrays directly.
    This is still more portable than full model pickle since only tree structures
    are pickled, not the entire model.
    
    Parameters:
        tree: sklearn DecisionTreeRegressor object
        
    Returns:
        Dictionary containing tree structure (base64-encoded pickle)
    """
    # Pickle the tree and encode as base64 for JSON compatibility
    tree_pickle = pickle.dumps(tree)
    tree_b64 = base64.b64encode(tree_pickle).decode("utf-8")
    
    return {"_pickle": tree_b64}


def tree_from_dict(tree_dict: Dict[str, Any]) -> DecisionTreeRegressor:
    """
    Reconstruct a sklearn DecisionTreeRegressor from a dictionary.
    
    Parameters:
        tree_dict: Dictionary containing tree structure (base64-encoded pickle)
        
    Returns:
        DecisionTreeRegressor object
        
    Raises:
        ValueError: If the tree dictionary is invalid or corrupted
        TypeError: If the unpickled object is not a DecisionTreeRegressor
    """
    if "_pickle" not in tree_dict:
        raise ValueError(
            "Invalid tree dictionary: missing '_pickle' key. "
            "The dictionary may be corrupted or in an unsupported format."
        )
    
    try:
        # Decode base64 and unpickle the tree
        tree_b64 = tree_dict["_pickle"]
        tree_pickle = base64.b64decode(tree_b64.encode("utf-8"))
        tree = pickle.loads(tree_pickle)
    except (binascii.Error, UnicodeDecodeError) as e:
        raise ValueError(
            f"Failed to decode tree data: {e}. "
            "The model file may be corrupted."
        ) from e
    except (pickle.PickleError, AttributeError, ImportError) as e:
        raise ValueError(
            f"Failed to unpickle tree: {e}. "
            "The model may have been saved with an incompatible version of sklearn."
        ) from e
    
    if not isinstance(tree, DecisionTreeRegressor):
        raise TypeError(
            f"Expected DecisionTreeRegressor, got {type(tree)}. "
            "The model file may be corrupted."
        )
    
    return tree


def numpy_to_list(obj: Any) -> Any:
    """
    Recursively convert numpy arrays and scalars to Python lists/values.
    
    Parameters:
        obj: Object that may contain numpy arrays
        
    Returns:
        Object with numpy arrays converted to lists
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_list(item) for item in obj]
    else:
        return obj

