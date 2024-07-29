import numpy as np
import jax.numpy as jnp
import manipulation_functions_for_hoppings as hop_funs
from manipulation_functions_for_hoppings import AtomicIndex, Hoppings
from typing import Dict, List
from functools import total_ordering
def arr(*arg):
    return np.array(*arg)
def dot(*arg):
    return np.matmul(*arg)
pi = np.pi
def inv(*arg):
    return np.linalg.inv(*arg)
def transpose(*arg):
    return np.transpose(*arg)
def exp(arg):
    return np.exp(arg)
def eigh(*arg):
    '''
    the eigenvalues are in asending order, the eigvectors are stored in columns
    '''
    return np.linalg.eigh(*arg)
def eigvalsh(*arg):
    '''
    the eigenvalues are in asending order
    '''
    return np.linalg.eigvalsh(*arg)
def int_floor(arg) -> int:
    return round(np.floor(arg))
def int_ceil(arg) -> int:
    return round(np.ceil(arg))

class Cell:
    def __init__(self, dirtocar: np.ndarray, prim_vecs: np.ndarray, num_sites: int, 
                 sitedir: List[np.ndarray]) -> None:
        self.dirtocar = dirtocar
        self.prim_vecs_dir = prim_vecs
        self.num_sites = num_sites
        self.sitedir = sitedir
    def sitedir_copy(self) -> list:
        return [pos.copy() for pos in self.sitedir]
    def _deep_copy(self) -> "Cell":
        return Cell(self.dirtocar.copy(), self.prim_vecs_dir.copy(),
                    self.num_sites, self.sitedir_copy())

class SingleParticleSystem:
    """
    hoppings is realized as a list of dictionaries
    """
    def __init__(self, cell: Cell, hoppings: Hoppings) -> None:
        self.cell = cell
        self.hoppings = hoppings
    def _deep_copy(self) -> "SingleParticleSystem":
        SingleParticleSystem(self.cell._deep_copy(), hop_funs.hop_copy(self.hoppings))
    def enlarge_cell(self, nmx:int, nmy:int) -> "SingleParticleSystem":
        num_s_prim = self.cell.num_sites
        num_s_ec = num_s_prim*nmx*nmy

        def atind_prim_to_ec(ind: AtomicIndex) -> AtomicIndex:
            remain = np.mod(ind.bravis, (nmx,nmy)) #remain is a np.ndarray
            i_ec = ind.sitelabel + num_s_prim*(nmy*remain[0] + remain[1])
            bravis_ec = (int_floor(ind.bravis[0]/nmx), int_floor(ind.bravis[1]/nmy))
            return AtomicIndex(i_ec, bravis_ec)
        def atind_ec_to_prim(ind: AtomicIndex) -> AtomicIndex:
            i_prim = np.mod(ind.sitelabel, num_s_prim)
            prim_cell_label = int_floor(ind.sitelabel/num_s_prim)
            bravis_prim = (int_floor(prim_cell_label/nmy) + ind.bravis[0]*nmx,
                           np.mod(prim_cell_label, nmy) + ind.bravis[1]*nmy)
            return AtomicIndex(i_prim, bravis_prim)
        def atind_trans(ind: AtomicIndex, origin: tuple) -> AtomicIndex:
            return AtomicIndex(ind.sitelabel, 
                               tuple((ind.bravis[i] + o for i, o in enumerate(origin))) #convert generator to tuple 
                               ) 
        sitedir_ec = []
        for i_ec in range(num_s_ec):
            atind_prim = atind_ec_to_prim(AtomicIndex(i_ec, (0,0)))
            i_prim = atind_prim.sitelabel
            site_i_dir = self.cell.sitedir[i_prim] +\
            dot(atind_prim.bravis, self.cell.prim_vecs_dir)
            sitedir_ec.append(site_i_dir)
        privec_ec = arr([nmx*self.cell.prim_vecs_dir[0], nmy*self.cell.prim_vecs_dir[1]])
        cell_ec = Cell(self.cell.dirtocar.copy(), privec_ec,
                       num_s_ec, sitedir_ec)
        hop_ec = [dict() for i in range(num_s_ec)]
        for i_ec in range(num_s_ec):
            origin_prim = atind_ec_to_prim(AtomicIndex(i_ec, (0,0)))
            i_prim = origin_prim.sitelabel # i_prim the primitive label for origin
            for end_prim in self.hoppings[i_prim]:
                hop_ec[i_ec][atind_prim_to_ec(atind_trans(end_prim, origin_prim.bravis))] =\
                self.hoppings[i_prim][end_prim]
        return SingleParticleSystem(cell_ec, hop_ec)

    def spin_duplicate(self) -> "SingleParticleSystem":
        """
        enlarge a spinless system by adding its time-reversal conjugate
        """
        sitedir_sd = self.cell.sitedir_copy() + self.cell.sitedir_copy()
        cell_sd = Cell(self.cell.dirtocar.copy(), self.cell.prim_vecs_dir.copy(),
                       2*self.cell.num_sites, sitedir_sd)
        hop_sd = hop_funs.hop_copy(self.hoppings) +\
              hop_funs.hop_spin_conjugate(self.hoppings)
        return SingleParticleSystem(cell_sd, hop_sd)

def sps_add_hop(sps:SingleParticleSystem, hop:Hoppings, operation_type = "pure_function") -> SingleParticleSystem:
    if operation_type == "pure_function":
        return SingleParticleSystem(sps.cell._deep_copy(), hop_funs.hop_add(sps.hoppings, hop))
    elif operation_type == "inplace":
        hop_funs.hop_add(sps.hoppings, hop, operation_type)
    else:
        raise ValueError("operation type error!")


class EigenSystem:
    def __init__(self, eigvals: np.ndarray, eigfuns: np.ndarray) -> None:
        if len(eigvals) != eigfuns.shape[1]:
            raise TypeError("eigvals and eigvecs have different numbers!")
        self.num_eigs = len(eigvals)
        self.eigvals = eigvals
        self.eigfuns = eigfuns
    def get_eigval_i(self, i: int):
        return self.eigvals[i]
    def get_eigfun_i(self, i:int):
        return self.eigfuns[:,i]
    @classmethod
    def from_tuple(cls, eigsys: tuple):
        return cls(eigsys[0], eigsys[1])

def cross_2d(a:np.ndarray, b:np.ndarray):
    if len(a) == 2 and len(b) == 2:
        return a[0]*b[1]-a[1]*b[0]
    elif len(a) == 1 and len(b) == 2:
        return arr([-b[1],b[0]])*a[0]
    elif len(a) == 2 and len(b) == 1:
        return arr([a[1],-a[0]])*b[0]
    else:
        return 0

def fracktocar(dirtocar:np.ndarray, privec:np.ndarray) -> np.ndarray:
    """
    take the k vector in fractional recipocal vectors to cartisian coordinates
    """
    a1 = dot(dirtocar,privec[0])
    a2 = dot(dirtocar,privec[1])
    vcell = cross_2d(a1,a2)
    g1 = 2*pi*cross_2d(a2,arr([1]))/np.abs(vcell)
    g2 = 2*pi*cross_2d(arr([1]),a1)/np.abs(vcell)
    return transpose(arr([g1,g2]))

def hamiltonian(sps:SingleParticleSystem, kfrac:np.ndarray) -> np.ndarray:
    '''
    construct Hamiltonian for sps on a certain k point(input kpoint in fractional coordinates of
    recipocal vectors)
    '''
    privec = sps.cell.prim_vecs_dir
    dirtocar = sps.cell.dirtocar
    num_sites = sps.cell.num_sites
    sitedir = sps.cell.sitedir
    hops = sps.hoppings
    kcar = dot(fracktocar(dirtocar, privec), arr(kfrac))
    hamiltonian = np.zeros((num_sites, num_sites), dtype=np.complex128)
    for i in range(num_sites):
        for ind in hops[i]:
            j = ind.sitelabel
            dsdir = dot(ind.bravis, privec) + sitedir[j] - sitedir[i]
            hamiltonian[j,i] += hops[i][ind]*exp(-1j*dot(kcar,dot(dirtocar, dsdir)))
    return hamiltonian

def bandstructure(sps: SingleParticleSystem, kpath: list):
    '''
    output the bandstructure of sps along kpath
    '''
    privec = sps.cell.prim_vecs_dir
    dirtocar = sps.cell.dirtocar
    ktocar = fracktocar(dirtocar,privec)
    kpoints = [arr(kpath[i-2]) + (arr(kpath[i])-arr(kpath[i-2]))*n/kpath[i-1] 
               for i in range(2, len(kpath), 2) for n in range(kpath[i-1])]
    l = 0
    tem = []
    for i in range(1, len(kpoints)):
        dk = dot(ktocar, kpoints[i]-kpoints[i-1])
        l += np.linalg.norm(dk)
        tem.append((l, kpoints[i]))
    kpoints = [(0.0, kpoints[0])] + tem
    tem = []
    for i in range(len(kpoints)):
        v = eigvalsh(hamiltonian(sps, kpoints[i][1]))
        tem.append([(kpoints[i][0], v[j]) for j in range(len(v))])
    return transpose(tem,(1,0,2))    

# region Custom Region
# def eigsys_dict(sps: SingleParticleSystem, kpts: list) -> Dict[tuple,EigenSystem]:
#     '''
#     create a typed dictionary that maps each kpoint to its eigsys(eigvals and eigfuns)
#     '''
#     return {kpt: EigenSystem.from_tuple(eigh(hamiltonian(sps, arr(kpt)))) 
#             for kpt in kpts}
#endregion

@total_ordering
class EigenState:
    '''
    contains the energy, k vector and bloch function for a eigenstate
    '''
    def __init__(self, kvec: np.ndarray, energy: float, bloch_fun: np.ndarray):
        self._energy = energy
        self._kvec = kvec
        self._bloch_fun = bloch_fun

    def __eq__(self, other):
        if isinstance(other, EigenState):
            return self.energy == other.energy
        else:
            raise TypeError("wrong type for comparison!")
    def __lt__(self, other):
        if isinstance(other, EigenState):
            return self.energy < other.energy
        else:
            raise TypeError("wrong type for comparison!")
    @property
    def energy(self) -> float:
        return self._energy
    @property
    def kvec(self) -> np.ndarray:
        return self._kvec
    @property
    def bloch_fun(self) -> np.ndarray:
        return self._bloch_fun
    def bloch_fun_renormalize(self, a):
        """
        rescale the bloch function to a times its original value 
        """
        self._bloch_fun *= a

def eigstate_flatten_sort(sps: SingleParticleSystem, kpts: list) -> List[EigenState]:
    eigstates = []
    for kpt in kpts:
        eigsys = EigenSystem.from_tuple(eigh(hamiltonian(sps, arr(kpt))))
        for i in range(eigsys.num_eigs):
            eigstates.append(EigenState(kpt, eigsys.get_eigval_i(i), eigsys.get_eigfun_i(i)))
    eigstates.sort()
    return eigstates
        








        