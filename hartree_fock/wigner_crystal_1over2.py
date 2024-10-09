import single_particle_class as spcl
#from single_particle_class import wavefunctions
import plot_functions as plt
import manipulation_functions_for_hoppings as hop_funs
import interaction
import seed as seed_generation
import hartree_fock_solvers
import numpy as np
from typing import Dict, List
import itertools
import argparse
import pickle

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

if __name__ == "__main__":
    theta = 4.0*(np.pi)/180
    a_m = 1/theta
    a1 = 1/theta*arr([np.sqrt(3)/2, 1/2])
    a2 = 1/theta*arr([-np.sqrt(3)/2, 1/2])
    privec = arr([[1,0], [0,1]])
    rsdirtocar = transpose(arr([a1, a2]))
    site_dir_coord = [arr([1/3, 2/3]), arr([2/3, 1/3])]
    cellprim = spcl.Cell(rsdirtocar, privec, 2, site_dir_coord)
    hoppings = read_hoppings('./hoppings.dat')
    #hoppings = hop_funs.hop_mul(hoppings, 0.01) #db
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float)
    parser.add_argument('--hfield', type=float)
    parser.add_argument('--hubbard_u', type=float) #0.2
    #parser.add_argument('cutoff', type=int) #6
    parser.add_argument('--scaling', type=float) #436 for epsilon=10
    parser.add_argument('--noise', type=float, default=0.0) #8.0
    parser.add_argument('--den_plots_suffix', type=str, default='')
    parser.add_argument('--output_suffix', type=str, default='')
    parser.add_argument('--write_restart', action='store_true')
    parser.add_argument('--read_restart', action='store_true')
    args = parser.parse_args()
    delta = args.delta #displacement field
    for i in range(cellprim.num_sites):
        hop_funs.hop_add_i(hoppings, i, {hop_funs.AtomicIndex(i,(0,0)): -(-1)**i*delta/2}, "inplace") #apply displacement field
    sps_prim = spcl.SingleParticleSystem(cellprim, hoppings)
    nmx = 4
    nmy = 4
    sps_ec = sps_prim.enlarge_cell(nmx, nmy)
    sps_sd = sps_ec.spin_duplicate()
    #sps_sd = (sps_ec.spin_duplicate()).apply_pbc() #periodic boundary condition
    hz_field = -(args.hfield)/2
    hop_funs.hop_apply_h(sps_sd.hoppings, 
                         arr([[0, 0, hz_field] for i in range(sps_sd.cell.num_sites//2)]), "inplace") #apply magnetic field
    
    #kline = [arr([0,0]), 20, arr([-1/3,2/3]), 40, arr([1/3,1/3]), 20, arr([0,0])]
    #bs = spcl.bandstructure(sps_ec, kline)
    #plt.list_plot(bs)
    
    vdd = interaction.truncated_coulomb(sps_sd.cell, 0.2, 6*a_m + 0.1, 363)
    #vdd = interaction.pbc_screened_coulomb(sps_sd.cell, args.hubbard_u, args.scaling, inf=24, shell=0.01, subtract_offset=False)
    #vdd = interaction.pbc_coulomb(sps_sd.cell, args.hubbard_u, args.scaling, inf=24, shell=0.01, subtract_offset=True) #sharp cutoff shell=0.01
    print("vdd constructed!")
    kgrid = hartree_fock_solvers.Kgrid((0,0),(3,3))
    controller = hartree_fock_solvers.Controller(200, 0.002, 0.5)
    seed = seed_generation.fmz_stripe_seed_honcomblattice(nmx,nmy)*100
    if args.read_restart:
        with open('./seed_restart.pkl', 'rb') as file:
            seed = pickle.load(file)
    sigma_new = hartree_fock_solvers.hartree_fock_solver(sps_sd, vdd, 1/8, kgrid, controller, seed, noise=args.noise,
                                             save_den_plots=True, save_output=True, saving_dir='./results_1over2',
                                             output_comment = f"delta = {args.delta}, hfield = {args.hfield}, hubbard_u = {args.hubbard_u}, scaling = {args.scaling}\n",
                                             den_plots_suffix=args.den_plots_suffix, output_suffix=args.output_suffix)
    if args.write_restart:
        with open('./seed_restart.pkl', 'wb') as file:
            pickle.dump(sigma_new, file)