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
import argparse

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

def read_hoppings(file_path) -> spcl.Hoppings:
    with open(file_path,"r") as file:
        num_sites = int(file.readline())
    hoppings = [{} for i in range(num_sites)]
    i = 0
    with open(file_path,"r") as file:
        for line in file:
            line.strip()
            entries = line.split()
            if len(entries) == 1:
                continue
            elif len(entries) == 0:
                i += 1
            else:
                hoppings[i][spcl.AtomicIndex(int(entries[0])-1,(int(entries[1]),int(entries[2])))] =\
                np.complex128(float(entries[3]),float(entries[4])) 
                     #translate into numbers, note that entries[0] is shifted to start from 0
    return(hoppings)

# def site_dir_coord(i: int) -> np.ndarray:
#     if i == 0:
#         return arr([1/3, 2/3])
#     else:
#         return arr([2/3, 1/3])

if __name__ == "__main__":
    theta = 4.0*(np.pi)/180
    a_m = 1/theta
    a1 = 1/theta*arr([np.sqrt(3)/2, 1/2])
    a2 = 1/theta*arr([-np.sqrt(3)/2, 1/2])
    privec = arr([[1,0], [0,1]])
    rsdirtocar = transpose(arr([a1, a2]))
    site_dir_coord = [arr([0, 0])]
    cellprim = spcl.Cell(rsdirtocar, privec, 1, site_dir_coord)
    t = -3.0
    hoppings = [{AtomicIndex(0,(1,0)):t,
                 AtomicIndex(0,(0,1)):t,
                 AtomicIndex(0,(1,1)):t,
                 AtomicIndex(0,(-1,0)):t,
                 AtomicIndex(0,(0,-1)):t,
                 AtomicIndex(0,(-1,-1)):t
                 }]
    sps_prim = spcl.SingleParticleSystem(cellprim, hoppings)
    nmx = 12
    nmy = 12
    sps_ec = sps_prim.enlarge_cell(nmx, nmy)
    sps_sd = (sps_ec.spin_duplicate()).apply_pbc()
    kline = [arr([0,0]), 20, arr([-1/3,2/3]), 40, arr([1/3,1/3]), 20, arr([0,0])]
    vdd = interaction.pbc_coulomb(sps_sd.cell, 0.2, 118, inf=24)
    kgrid = hartree_fock_solvers.Kgrid((0,0),(1,1))
    controller = hartree_fock_solvers.Controller(1000, 0.002, 0.5)
    seed = seed_generation.fmz_honcomb_seed_trilattice(nmx,nmy)*100
    sigma_new = hartree_fock_solvers.hartree_fock_solver(sps_sd, vdd, 1/3, kgrid, controller, seed, noise=0.0,
                                             save_den_plots=True, save_output=True, saving_dir='./results',
                                             output_comment = "\n")
    hop_scf = hop_funs.hop_add(sps_sd.hoppings, sigma_new)
    def hop_vs_distance(cell: spcl.Cell, hop: hop_funs.Hoppings, i: int):
        sitedir = cell.sitedir
        privec = cell.prim_vecs_dir
        dirtocar = cell.dirtocar
        num_sites = cell.num_sites
        hop_list = [(np.linalg.norm(dot(dirtocar, sitedir[ind.sitelabel] - sitedir[i] + dot(ind.bravis,privec))),
                    np.abs(hop[i][ind])) for ind in hop[i]]
        hop_list.sort(key=lambda x: x[0])
        return hop_list
    hop_list = hop_vs_distance(sps_sd.cell, hop_scf, 14) #14
    # print(hop_scf[0][AtomicIndex(1,(0,0))])
    # print(hop_scf[0][AtomicIndex(6,(0,0))])
    # print(hop_scf[0][AtomicIndex(7,(0,0))])
    # print(hop_scf[0][AtomicIndex(5,(0,0))])
    # print(hop_scf[0][AtomicIndex(35,(0,0))])
    # print(hop_scf[0][AtomicIndex(30,(0,0))])
    #print(hop_list)
    plt.list_plot(hop_list[1:])
