import numpy as np
from typing import Dict, List, Tuple

# region Custom Region
# class AtomicIndex:
#     """
#     label an arbitrary site in arbitrary cell, hashable, comparable, immutable
#     """
#     def __init__(self, sitelabel: int, bravis: tuple) -> None:
#         self.sitelabel = sitelabel
#         self.bravis = bravis
#     def __eq__(self, other):
#         if isinstance(other, AtomicIndex):
#             return hash((self.sitelabel, ) + self.bravis) ==\
#             hash((other.sitelabel, ) + other.bravis)
#         else:
#             return False
#     def __hash__(self) -> int:
#         return hash((self.sitelabel, ) + self.bravis) 
# endregion
def arr(*arg):
    return np.array(*arg)

class AtomicIndex:
    """
    label an arbitrary site in arbitrary cell, hashable, comparable, immutable
    """
    def __init__(self, sitelabel: int, bravis) -> None:
        if isinstance(bravis, tuple):
            self._val = (sitelabel, bravis)
        elif isinstance(bravis, np.ndarray):
            self._val = (sitelabel, tuple(bravis))
        else:
            raise TypeError("wrong type for bravis!")

    @property
    def sitelabel(self) -> int:
        return self._val[0]
    @property
    def bravis(self) -> np.ndarray:
        return np.array(self._val[1]) #return the bravis vectors as array
    
    def __eq__(self, other):
        if isinstance(other, AtomicIndex):
            return self._val == other._val
        else:
            return False
    def __hash__(self) -> int:
        return hash(self._val)
    
    @classmethod
    def from_tuple(cls, val: Tuple[int, Tuple[int,...]]) -> "AtomicIndex":
        return AtomicIndex(val[0], val[1])
 
Hoppings = List[Dict[AtomicIndex, np.complex128]] #alias class name "Hoppings"

def hop_copy(hop: Hoppings) -> Hoppings:
    return [{ind: hopi[ind] for ind in hopi} for hopi in hop]

def hop_add(hop0:Hoppings, hop:Hoppings, operation_type = "pure_function") -> Hoppings:
    """
    add the second argument to the first one, 
    "operation_type" controls whether the operation is inplace
    """
    if operation_type == "pure_function":
        hop0_copy = hop_copy(hop0)
        num_sites = len(hop0)
        for i in range(num_sites):
            for ind in hop[i]:
                if ind in hop0[i]:
                    hop0_copy[i][ind] += hop[i][ind]
                else:
                    hop0_copy[i][ind] = hop[i][ind]
        return hop0_copy
        
    elif operation_type == "inplace":
        num_sites = len(hop0)
        for i in range(num_sites):
            for ind in hop[i]:
                if ind in hop0[i]:
                    hop0[i][ind] += hop[i][ind]
                else:
                    hop0[i][ind] = hop[i][ind]
    else:
        raise ValueError("operation type error!")

def hop_add_i(hop0:Hoppings, i:int, hopi:Dict[AtomicIndex, np.complex128],
               operation_type = "pure_function") -> Hoppings:
    """
    add the second argument to the first one, "out" controls the output
    """
    if operation_type == "pure_function":
        hop0_copy = hop_copy(hop0)
        for ind in hopi:
            if ind in hop0_copy[i]:
                hop0_copy[i][ind] += hopi[ind]
            else:
                hop0_copy[i][ind] = hopi[ind]
        return hop0_copy
    elif operation_type == "inplace":
        for ind in hopi:
            if ind in hop0[i]:
                hop0[i][ind] += hopi[ind]
            else:
                hop0[i][ind] = hopi[ind]
    else:
        raise ValueError("operation type error!")
    
def hop_apply_h(hop: Hoppings, hfields: np.ndarray, operation_type = "pure_function") -> Hoppings:
    sx = arr([[0,1],[1,0]])
    sy = arr([[0,-1j],[1j,0]])
    sz = arr([[1,0],[0,-1]])
    sv = [sx, sy, sz]
    num_sites = len(hop)
    if len(hfields) != num_sites/2:
        raise TypeError("imcompactable size for hfields")
    elif operation_type == "inplace":
        onsiteh_mats = [sum(onsiteh[j]*sv[j] for j in range(3)) for i, onsiteh in enumerate(hfields)]
        for i, mat in enumerate(onsiteh_mats):
            hop_add_i(hop, i, {AtomicIndex(i,(0,0)):mat[0,0]}, "inplace")
            hop_add_i(hop, i, {AtomicIndex(i+num_sites//2,(0,0)):mat[1,0]}, "inplace")
            hop_add_i(hop, i+num_sites//2, {AtomicIndex(i,(0,0)):mat[0,1]}, "inplace")
            hop_add_i(hop, i+num_sites//2, {AtomicIndex(i+num_sites//2,(0,0)):mat[1,1]}, "inplace")
    elif operation_type == "pure_function":
        hop_copy = hop_copy(hop)
        onsiteh_mats = [sum(onsiteh[j]*sv[j] for j in range(3)) for i, onsiteh in enumerate(hfields)]
        for i, mat in enumerate(onsiteh_mats):
            hop_add_i(hop_copy, i, {AtomicIndex(i,(0,0)):mat[0,0]}, "inplace")
            hop_add_i(hop_copy, i, {AtomicIndex(i+num_sites//2,(0,0)):mat[1,0]}, "inplace")
            hop_add_i(hop_copy, i+num_sites//2, {AtomicIndex(i,(0,0)):mat[0,1]}, "inplace")
            hop_add_i(hop_copy, i+num_sites//2, {AtomicIndex(i+num_sites//2,(0,0)):mat[1,1]}, "inplace")
        return hop_copy
    else:
        raise TypeError("operation type error!")

def hop_mul(hop:Hoppings, a) -> Hoppings:
    return [{ind: a*hopi[ind] for ind in hopi} for hopi in hop]

def hop_norm(hop:Hoppings):
    return np.sqrt(sum(sum(np.conjugate(h)*h for h in hopi.values()) for hopi in hop))

def hop_spin_conjugate(hop:Hoppings) -> Hoppings:
    num_sites = len(hop)
    return [{AtomicIndex(ind.sitelabel + num_sites, ind.bravis) : np.conjugate(hopi[ind]) 
             for ind in hopi} for hopi in hop]