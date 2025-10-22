import torch
import numpy as np
from typing import Any

def equal_dicts(dict1, dict2):
    if dict1 is None or dict2 is None:
        return False

    for key in dict1.keys():
        if isinstance(dict1[key], torch.Tensor):
            if not torch.equal(dict1[key], dict2[key]):
                return False
        elif isinstance(dict1[key], np.ndarray):
            if not np.array_equal(dict1[key], dict2[key]):
                return False
        # elif isinstance(dict1[key], list):
        #     return equal_lists(dict1[key], dict2[key])
        else:
            if key not in dict2.keys():
                return False
            if dict1[key] != dict2[key]:
                return False
    return True

#
# def equal_lists(list1, list2):
#     if len(list1) != len(list1):
#         return False
#
#     for i in range(len(list1)):
#         if isinstance(list1[i], torch.Tensor):
#             if not torch.equal(list1[i], list2[i]):
#                 return False
#         elif isinstance(list1[i], np.ndarray):
#             if not np.array_equal(list1[i], list2[i]):
#                 return False
#         elif isinstance(list1[i], list):
#             return equal_lists(list1[i], list2[i])
#         elif isinstance(list1[i], dict):
#             return equal_dicts(list1[i], list2[i])
#         else:
#             if list1[i] != list2[i]:
#                 return False
#     return True


class EasyDict(dict):

    @property
    def __name__(self):
        return self.__class__.__name__

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]
