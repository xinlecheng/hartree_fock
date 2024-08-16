import numpy as np
from typing import List, Dict, Tuple
from manipulation_functions_for_hoppings import AtomicIndex
import single_particle_class

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

density_density_type = List[Dict[AtomicIndex, np.complex128]]

def truncated_coulomb(cell_sd: single_particle_class.Cell, hubbard_u, cutoff = 0, scaling = 1) -> density_density_type:
    """
    cutoff is measured in cartesian metric
    """
    num_sites = cell_sd.num_sites
    sitedir = cell_sd.sitedir
    dirtocar = cell_sd.dirtocar
    privec = cell_sd.prim_vecs_dir
    num_bravis_x = int_ceil(2/np.sqrt(3)*cutoff/np.linalg.norm(dot(dirtocar,privec[0])))
    num_bravis_y = int_ceil(2/np.sqrt(3)*cutoff/np.linalg.norm(dot(dirtocar,privec[1])))
    bravis_range = [(bravis_x, bravis_y) 
                    for bravis_x in range(-num_bravis_x, num_bravis_x+1) for bravis_y in range(-num_bravis_y, num_bravis_y+1)]
    a_m = np.linalg.norm(dirtocar[:,0]) #the moire lattice constant
    def vr(r):
        if r > 0.1*a_m:
            return 1/np.sqrt(r**2 + (0.1*a_m)**2)*scaling
        else:
            return hubbard_u*scaling
    vint = [dict() for i in range(num_sites)]
    for i in range(num_sites):
        vint[i] = {AtomicIndex(j, bravis): vr(np.linalg.norm(dot(dirtocar, dot(bravis, privec) + sitedir[j] - sitedir[i])))
                    for j in range(num_sites) for bravis in bravis_range 
                    if np.linalg.norm(dot(dirtocar, dot(bravis, privec) + sitedir[j] - sitedir[i])) <= cutoff}
    return vint

def pbc_coulomb(cell_sd: single_particle_class.Cell, hubbard_u, scaling = 1, 
                inf = 24, shell = 6, subtract_offset=True) -> density_density_type:
    '''
    infinitely long range coulomb with PBC,
    shell_width controls the softness of cutoff
    '''
    num_sites = cell_sd.num_sites
    sitedir = cell_sd.sitedir
    dirtocar = cell_sd.dirtocar
    privec = cell_sd.prim_vecs_dir
    a_m = np.linalg.norm(dirtocar[:,0]) #the moire lattice constant
    
    cutoff = (inf + shell)*a_m + 0.1 # in practice need to choose a value for inf, for instance 24
    inner_cutoff = inf*a_m
    decay_width = shell*a_m/2 # decay width=shell_width/2,  can also be tuned
    num_bravis_x = int_ceil(2/np.sqrt(3)*cutoff/np.linalg.norm(dot(dirtocar,privec[0])))
    num_bravis_y = int_ceil(2/np.sqrt(3)*cutoff/np.linalg.norm(dot(dirtocar,privec[1])))
    bravis_range = [(bravis_x, bravis_y) 
                    for bravis_x in range(-num_bravis_x, num_bravis_x+1) for bravis_y in range(-num_bravis_y, num_bravis_y+1)]
    def vr(r):
        if r > inner_cutoff:
            return 1/np.sqrt(r**2 + (0.1*a_m)**2)*scaling*exp(-(r-inner_cutoff)/decay_width)
        elif r > 0.1*a_m:
            return 1/np.sqrt(r**2 + (0.1*a_m)**2)*scaling
        else:
            return hubbard_u*scaling
    if subtract_offset:
        offset = sum(vr(np.linalg.norm(dot(dirtocar, dot(bravis, privec))))
                    for bravis in bravis_range 
                    if np.linalg.norm(dot(dirtocar, dot(bravis, privec))) <= cutoff) - hubbard_u*scaling #exclude onsite interaction
    else:
        offset = 0.0
    print("offset = ", offset) #db
    vint = [dict() for i in range(num_sites)]
    for i in range(num_sites):
        vint[i] = {AtomicIndex(j, (0,0)): sum(vr(np.linalg.norm(dot(dirtocar, dot(bravis, privec) + sitedir[j] - sitedir[i])))
                    for bravis in bravis_range 
                    if np.linalg.norm(dot(dirtocar, dot(bravis, privec) + sitedir[j] - sitedir[i])) <= cutoff) - offset
                    for j in range(num_sites) }
    return vint