import numpy as np
from manipulation_functions_for_hoppings import AtomicIndex

def int_floor(arg) -> int:
    return round(np.floor(arg))
def int_ceil(arg) -> int:
    return round(np.ceil(arg))

def fmz_unif_seed_honcomblattice(nmx:int, nmy:int) -> np.ndarray:
    """
    seed that generate ferroz uniform configuration on honeycomb lattice
    """
    return np.array([[0,0,-1] for i in range(nmx*nmy*2)])
def fmz_unif_seed_trilattice(nmx:int, nmy:int) -> np.ndarray:
    """
    seed that generate ferroz uniform configuration on triangular lattice
    """
    return np.array([[0,0,-1] for i in range(nmx*nmy)])
def fmx_unif_seed_honcomblattice(nmx:int, nmy:int) -> np.ndarray:
    """
    seed that generate ferroz uniform configuration on honeycomb lattice
    """
    return np.array([[-1,0,0] for i in range(nmx*nmy*2)])
def fmz_3over5_seed_honcomblattice(nmx:int, nmy:int) -> np.ndarray:
    '''
    seed that generate ferroz configuration for 3/5 filling on honeycomb lattice
    '''
    num_s_prim = 2
    def atind_ec_to_prim(ind: AtomicIndex) -> AtomicIndex:
            i_prim = np.mod(ind.sitelabel, num_s_prim)
            prim_cell_label = int_floor(ind.sitelabel/num_s_prim)
            bravis_prim = (int_floor(prim_cell_label/nmy) + ind.bravis[0]*nmx,
                           np.mod(prim_cell_label, nmy) + ind.bravis[1]*nmy)
            return AtomicIndex(i_prim, bravis_prim)
    seed = []
    for i in range(nmx*nmy*2):
        atind_prim = atind_ec_to_prim(AtomicIndex(i,(0,0)))
        if atind_prim.sitelabel == 0 and\
        np.mod(atind_prim.bravis[0],2) == 0 and\
        np.mod(atind_prim.bravis[1]-atind_prim.bravis[0]//2, 5) <= 2:
             seed.append([0, 0, -1])
        elif atind_prim.sitelabel == 0 and\
        np.mod(atind_prim.bravis[0],2) == 1 and\
        np.mod(atind_prim.bravis[1]-(atind_prim.bravis[0]-5)//2, 5) <= 2:
             seed.append([0, 0, -1])
        else:
             seed.append([0, 0, 0])
    return np.array(seed)
 
def fmz_honcomb_seed_honcomblattice(nmx:int, nmy:int) -> np.ndarray:
    """
    seed that generate ferroz honeycomb configuration on honeycomb lattice(1/4 filling without spin)
    """
    num_s_prim = 2
    def atind_ec_to_prim(ind: AtomicIndex) -> AtomicIndex:
            i_prim = np.mod(ind.sitelabel, num_s_prim)
            prim_cell_label = int_floor(ind.sitelabel/num_s_prim)
            bravis_prim = (int_floor(prim_cell_label/nmy) + ind.bravis[0]*nmx,
                           np.mod(prim_cell_label, nmy) + ind.bravis[1]*nmy)
            return AtomicIndex(i_prim, bravis_prim)
    seed = []
    for i in range(nmx*nmy*2):
        atind_prim = atind_ec_to_prim(AtomicIndex(i,(0,0)))
        if atind_prim.sitelabel == 0 and np.mod(atind_prim.bravis[1],3) != np.mod(2*atind_prim.bravis[0]+2,3):
             seed.append([0, 0, -1])
        else:
             seed.append([0, 0, 0])
    return np.array(seed)
def afmz_honcomb_seed_honcomblattice(nmx: int, nmy:int) -> np.ndarray:
    """
    seed that generate antiferroz honeycomb configuration on honeycomb lattice(1/4 filling without spin)
    """
    num_s_prim = 2
    def atind_ec_to_prim(ind: AtomicIndex) -> AtomicIndex:
            i_prim = np.mod(ind.sitelabel, num_s_prim)
            prim_cell_label = int_floor(ind.sitelabel/num_s_prim)
            bravis_prim = (int_floor(prim_cell_label/nmy) + ind.bravis[0]*nmx,
                           np.mod(prim_cell_label, nmy) + ind.bravis[1]*nmy)
            return AtomicIndex(i_prim, bravis_prim)
    seed = []
    for i in range(nmx*nmy*2):
        atind_prim = atind_ec_to_prim(AtomicIndex(i,(0,0)))
        if atind_prim.sitelabel == 0 and np.mod(atind_prim.bravis[1],3) == np.mod(2*atind_prim.bravis[0],3):
             seed.append([0, 0, -1])
        elif atind_prim.sitelabel == 0 and np.mod(atind_prim.bravis[1],3) == np.mod(2*atind_prim.bravis[0]+1, 3):
             seed.append([0, 0, 1])
        else:
             seed.append([0, 0, 0])
    return np.array(seed)
def fmz_honcomb_seed_trilattice(nmx:int, nmy:int) -> np.ndarray:
    """
    seed that generate ferroz honeycomb configuration on triangular lattice(1/4 filling without spin)
    """
    num_s_prim = 1
    def atind_ec_to_prim(ind: AtomicIndex) -> AtomicIndex:
            i_prim = np.mod(ind.sitelabel, num_s_prim)
            prim_cell_label = int_floor(ind.sitelabel/num_s_prim)
            bravis_prim = (int_floor(prim_cell_label/nmy) + ind.bravis[0]*nmx,
                           np.mod(prim_cell_label, nmy) + ind.bravis[1]*nmy)
            return AtomicIndex(i_prim, bravis_prim)
    seed = []
    for i in range(nmx*nmy):
        atind_prim = atind_ec_to_prim(AtomicIndex(i,(0,0)))
        if atind_prim.sitelabel == 0 and np.mod(atind_prim.bravis[1],3) != np.mod(2*atind_prim.bravis[0]+2,3):
             seed.append([0, 0, -1])
        else:
             seed.append([0, 0, 0])
    return np.array(seed)
def fmz_stripe_seed_honcomblattice(nmx:int, nmy:int) -> np.ndarray:
    """
    seed that generate ferroz stripe configuration on honeycomb lattice(1/3 filling without spin) 
    """
    num_s_prim = 2
    def atind_ec_to_prim(ind: AtomicIndex) -> AtomicIndex:
            i_prim = np.mod(ind.sitelabel, num_s_prim)
            prim_cell_label = int_floor(ind.sitelabel/num_s_prim)
            bravis_prim = (int_floor(prim_cell_label/nmy) + ind.bravis[0]*nmx,
                           np.mod(prim_cell_label, nmy) + ind.bravis[1]*nmy)
            return AtomicIndex(i_prim, bravis_prim)
    seed = []
    for i in range(nmx*nmy*2):
        atind_prim = atind_ec_to_prim(AtomicIndex(i,(0,0)))
        if atind_prim.sitelabel == 0 and np.mod(atind_prim.bravis[0],2) == 0:
             seed.append([0, 0, -1])
        else:
             seed.append([0, 0, 0])
    return np.array(seed)
def afmz_stripe_seed_honcomblattice(nmx:int, nmy:int) -> np.ndarray:
    """
    seed that generate ferroz stripe configuration on honeycomb lattice(1/3 filling without spin) 
    """
    num_s_prim = 2
    def atind_ec_to_prim(ind: AtomicIndex) -> AtomicIndex:
            i_prim = np.mod(ind.sitelabel, num_s_prim)
            prim_cell_label = int_floor(ind.sitelabel/num_s_prim)
            bravis_prim = (int_floor(prim_cell_label/nmy) + ind.bravis[0]*nmx,
                           np.mod(prim_cell_label, nmy) + ind.bravis[1]*nmy)
            return AtomicIndex(i_prim, bravis_prim)
    seed = []
    for i in range(nmx*nmy*2):
        atind_prim = atind_ec_to_prim(AtomicIndex(i,(0,0)))
        if atind_prim.sitelabel == 0 and np.mod(atind_prim.bravis[0],2) == 0 and np.mod(atind_prim.bravis[1],2) == 0:
             seed.append([0, 0, -1])
        elif atind_prim.sitelabel == 0 and np.mod(atind_prim.bravis[0],2) == 0 and np.mod(atind_prim.bravis[1],2) == 1:
             seed.append([0, 0, 1])
        else:
             seed.append([0, 0, 0])
    return np.array(seed)
def fmz_stripe_seed_trilattice(nmx:int, nmy:int) -> np.ndarray:
    """
    seed that generate ferroz stripe configuration on triangular lattice(1/3 filling without spin) 
    """
    num_s_prim = 1
    def atind_ec_to_prim(ind: AtomicIndex) -> AtomicIndex:
            i_prim = np.mod(ind.sitelabel, num_s_prim)
            prim_cell_label = int_floor(ind.sitelabel/num_s_prim)
            bravis_prim = (int_floor(prim_cell_label/nmy) + ind.bravis[0]*nmx,
                           np.mod(prim_cell_label, nmy) + ind.bravis[1]*nmy)
            return AtomicIndex(i_prim, bravis_prim)
    seed = []
    for i in range(nmx*nmy):
        atind_prim = atind_ec_to_prim(AtomicIndex(i,(0,0)))
        if atind_prim.sitelabel == 0 and np.mod(atind_prim.bravis[0],2) == 0:
             seed.append([0, 0, -1])
        else:
             seed.append([0, 0, 0])
    return np.array(seed)

def fmz_noise_honcomblattice(nmx:int, nmy:int) -> np.ndarray:
    num_sites = 4*nmx*nmy
    return np.array([[0, 0, np.random.uniform(-1,0)] for i in range(num_sites//2)])
