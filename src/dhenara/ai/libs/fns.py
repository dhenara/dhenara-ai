import collections


def generic_obj_to_dict(obj):
    """
    Recursively convert an object to a dictionary.
    Handles lists, dictionaries, mappingproxy, protobuf messages, and objects with __dict__ attribute.
    """
    if isinstance(obj, dict):
        # If obj is a dictionary, apply the function to each value
        return {key: generic_obj_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, collections.abc.Mapping):
        # If obj is a mapping (including mappingproxy), convert to a dict
        return {key: generic_obj_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # If obj is a list, apply the function to each element
        return [generic_obj_to_dict(element) for element in obj]
    elif hasattr(obj, "__dict__"):
        # If obj is an object with a dictionary attribute, convert it
        return generic_obj_to_dict(obj.__dict__)
    else:
        # If obj is a basic type (int, str, etc.), return it as is
        return obj
