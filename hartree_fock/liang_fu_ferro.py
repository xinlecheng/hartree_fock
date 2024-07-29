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
    theta = 4.0*(np.pi)/180
    a_m = 1/theta
    a1 = 1/theta*arr([np.sqrt(3)/2, 1/2])
    a2 = 1/theta*arr([-np.sqrt(3)/2, 1/2])
    privec = arr([[1,0], [0,1]])
    rsdirtocar = transpose(arr([a1, a2]))
    site_dir_coord = [arr([1/3, 2/3]), arr([2/3, 1/3])]
    num_sites = 2
    cellprim = spcl.Cell(rsdirtocar, privec, num_sites, site_dir_coord)
    t_1 = -1.0
    delta = 0.0/2
    t_so = -0.30
    t_3 = 0.13
    alpha = 2*pi/3
    hoppings = [{AtomicIndex(0,(0,0)):delta,
                AtomicIndex(1,(0,0)):t_1,
                AtomicIndex(1,(0,1)):t_1,
                AtomicIndex(1,(-1,0)):t_1,
                AtomicIndex(0,(1,0)):exp(1j*alpha)*t_so,
                AtomicIndex(0,(1,1)):exp(-1j*alpha)*t_so,
                AtomicIndex(0,(0,1)):exp(1j*alpha)*t_so,
                AtomicIndex(0,(-1,0)):exp(-1j*alpha)*t_so,
                AtomicIndex(0,(-1,-1)):exp(1j*alpha)*t_so,
                AtomicIndex(0,(0,-1)):exp(-1j*alpha)*t_so,
                AtomicIndex(1,(1,1)):t_3,
                AtomicIndex(1,(-1,1)):t_3,
                AtomicIndex(1,(-1,-1)):t_3
                },
                {AtomicIndex(1,(0,0)):-delta,
                AtomicIndex(0,(0,0)):t_1,
                AtomicIndex(0,(0,-1)):t_1,
                AtomicIndex(0,(1,0)):t_1,
                AtomicIndex(1,(-1,0)):exp(1j*alpha)*t_so,
                AtomicIndex(1,(-1,-1)):exp(-1j*alpha)*t_so,
                AtomicIndex(1,(0,-1)):exp(1j*alpha)*t_so,
                AtomicIndex(1,(1,0)):exp(-1j*alpha)*t_so,
                AtomicIndex(1,(1,1)):exp(1j*alpha)*t_so,
                AtomicIndex(1,(0,1)):exp(-1j*alpha)*t_so,
                AtomicIndex(0,(-1,-1)):t_3,
                AtomicIndex(0,(1,-1)):t_3,
                AtomicIndex(0,(1,1)):t_3
                }
                ]
    sps_prim = spcl.SingleParticleSystem(cellprim, hoppings)
    nmx = 3
    nmy = 3
    sps_ec = sps_prim.enlarge_cell(nmx, nmy)
    sps_sd = sps_ec.spin_duplicate()
    kline = [arr([0,0]), 20, arr([-1/3,2/3]), 40, arr([1/3,1/3]), 20, arr([0,0])]
   
    #plt.list_plot(spcl.bandstructure(sps_prim, kline))
    vdd = interaction.truncated_coulomb(sps_sd.cell, 0.2, 0*a_m + 0.1, 50)
    kgrid = hartree_fock_solvers.Kgrid((0,0),(4,4))
    controller = hartree_fock_solvers.Controller(1000, 0.002, 0.5)
    seed = seed_generation.fmz_unif_seed_honcomblattice(nmx,nmy)*100
    print(hartree_fock_solvers.hartree_fock_solver(
        sps_sd, vdd, 1/4, kgrid, controller, seed, noise = 2.0,
        save_den_results=True, save_output=True, saving_dir='./results/'))