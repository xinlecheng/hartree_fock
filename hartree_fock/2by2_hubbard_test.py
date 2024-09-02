import single_particle_class as spcl
#from single_particle_class import wavefunctions
import plot_functions as plt
import manipulation_functions_for_hoppings as hop_funs
from manipulation_functions_for_hoppings import AtomicIndex
import interaction
import seed as seed_generation
import hartree_fock_solvers
import numpy as np
from typing import Dict, List
import itertools

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
    the eigenvalues are in asending order
    '''
    return np.linalg.eigh(*arg)
def eigvalsh(*arg):
    '''
    the eigenvalues are in asending order
    '''
    return np.linalg.eigvalsh(*arg)

if __name__ == "__main__":
    a1 = arr([1, 0])
    a2 = arr([0, 1])
    privec = arr([[1,0], [0,1]])
    rsdirtocar = transpose(arr([a1, a2]))
    site_dir_coord = [arr([0, 0])]
    num_sites = 1
    cellprim = spcl.Cell(rsdirtocar, privec, num_sites, site_dir_coord)
    t = -1/2
    U = 0.5
    hoppings = [{AtomicIndex(0,(0,0)):0,
                AtomicIndex(0,(1,0)):t,
                AtomicIndex(0,(-1,0)):t,
                AtomicIndex(0,(0,1)):t,
                AtomicIndex(0,(0,-1)):t,
                }]
    sps_prim = spcl.SingleParticleSystem(cellprim, hoppings)
    nmx = 2
    nmy = 2
    sps_ec = sps_prim.enlarge_cell(nmx, nmy)
    sps_sd = sps_ec.spin_duplicate()
    vdd = interaction.truncated_coulomb(sps_sd.cell, 1, 0+0.1, U)
    kgrid = hartree_fock_solvers.Kgrid((0,0),(1,1))
    controller = hartree_fock_solvers.Controller(1000, 0.0002, 0.5)
    seed = seed_generation.fmz_unif_seed_trilattice(nmx,nmy)*0
    sigma_new = hartree_fock_solvers.hartree_fock_solver(
       sps_sd, vdd, 1/2, kgrid, controller, seed, noise = 3.0,
       save_den_plots=True, save_output=True, saving_dir='./results/', den_plots_suffix='_test', output_suffix='_1_1')
    #plt.list_plot(spcl.bandstructure(spcl.sps_add_hop(sps_sd, sigma_new), kline), aspect_ratio=1/6)
    print("calculation complete!")