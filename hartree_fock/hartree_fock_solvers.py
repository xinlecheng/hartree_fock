import numpy as np
import manipulation_functions_for_hoppings as hop_funs
import single_particle_class as spcl
from manipulation_functions_for_hoppings import AtomicIndex
import interaction as interaction
from typing import List, Dict, Tuple
import itertools
import plot_functions
import os
import io_fun

# region Custom Region
def arr(*arg):
    return np.array(*arg)
def dot(*arg):
    return np.matmul(*arg)
pi = np.pi
def inv(*arg):
    return np.linalg.inv(*arg)
def conjugate(*arg):
    return np.conjugate(*arg)
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
def int_floor(arg) -> int:
    return round(np.floor(arg))
def int_ceil(arg) -> int:
    return round(np.ceil(arg))
#endregion

class Controller:
    def __init__(self, ite_max: int, cvg_crit: np.float128, mixing):
        self.ite_max = ite_max
        self.converge_crit = cvg_crit
        self.mixing = mixing

class Kgrid:
    def __init__(self, offset: tuple, enlargment:tuple):
        self.offset = offset
        self.enlargement = enlargment

def self_energy_init(sps0:spcl.SingleParticleSystem, vdd:interaction.density_density_type,
                     filling:np.float128, kgrid: Kgrid, seed, noise = 0.0):
    privec = sps0.cell.prim_vecs_dir
    dirtocar = sps0.cell.dirtocar
    num_sites = sps0.cell.num_sites
    sitedir = sps0.cell.sitedir

    ktocar = spcl.fracktocar(dirtocar, privec)
    inner_dsdir_kfrac = dot(transpose(dirtocar), ktocar)

    kpts = [arr([i/kgrid.enlargement[0]+kgrid.offset[0], j/kgrid.enlargement[1]+kgrid.offset[1]]) 
             for i in range(kgrid.enlargement[0]) 
             for j in range(kgrid.enlargement[1])]
    num_kpts = len(kpts)
    num_ele = round(num_sites*num_kpts*filling)

    if isinstance(seed, np.ndarray):
        if len(seed) == num_sites:
            sigma_0: hop_funs.Hoppings = [{AtomicIndex(i,(0,0)):onsitev} for i, onsitev in enumerate(seed)]
        elif len(seed) == num_sites/2:
            sigma_0 = [dict() for i in range(num_sites)]
            hop_funs.hop_apply_h(sigma_0, seed, "inplace")
        else:
            raise(TypeError("wrong seed type!"))
    else:
        sigma_0 = seed #hop_funs.hop_add(seed, hop_funs.hop_mul(sps0, -1)) #restart option

    eigen_states = spcl.eigstate_flatten_sort(spcl.sps_add_hop(sps0, sigma_0), kpts)
    #print(float(eigen_states[0].energy)) #db
    #print(dot(np.conjugate(eigen_states[1].bloch_fun),eigen_states[2].bloch_fun)) #db
    for eig in eigen_states:
        eig.bloch_fun_renormalize(1/np.sqrt(num_kpts)) #renormalize the bloch function for enlarged PBC
    #occ_states = eigen_states[:num_ele] #occupied eigenstates
    occ_states = eigen_states[:num_ele]
    #print([eigen_states[i].energy for i in range(num_ele+1)]) #db
    #print([float(eig.energy) for eig in eigen_states])#db
    den = [sum(np.conjugate(occ.bloch_fun)[i]*occ.bloch_fun[i] 
                for occ in occ_states) for i in range(num_sites)]
    #print(den) #db
    #plot_functions.den_plot(sps0.cell, den[:num_sites//2]) #db
    sigma_tem = [dict() for i in range(num_sites)]
    for i in range(num_sites):
        vddi = vdd[i]
        hop_funs.hop_add_i(sigma_tem, i,
                            {AtomicIndex(i,(0,0)): sum(vddi[ind]*den[ind.sitelabel] for ind in vddi)},
                            "inplace") # hartree term
        hop_funs.hop_add_i(sigma_tem, i,
                            {ind: -sum(vddi[ind]*np.conjugate(occ.bloch_fun[i])*occ.bloch_fun[ind.sitelabel]*\
                                        exp(1j*dot((dot(ind.bravis,privec) + sitedir[ind.sitelabel] - sitedir[i]), dot(inner_dsdir_kfrac, occ.kvec))) 
                                        for occ in occ_states) for ind in vddi},
                            "inplace") # fock terms
    noise_mats = np.random.uniform(-1, 1, (num_sites//2, 3))*noise #random magnetic fields as noise
    hop_funs.hop_apply_h(sigma_tem, noise_mats, "inplace")
    return sigma_tem

def gap_correction(cell:spcl.Cell, vdd:interaction.density_density_type, excit:spcl.EigenState, hole:spcl.EigenState) -> float:
    privec = cell.prim_vecs_dir
    dirtocar = cell.dirtocar
    sitedir = cell.sitedir
    ktocar = spcl.fracktocar(dirtocar, privec)
    inner_dsdir_kfrac = dot(transpose(dirtocar), ktocar)
    def ele_hole_int_partial(excit:spcl.EigenState, hole:spcl.EigenState, 
                            vddi:Dict[AtomicIndex, np.complex128], ind: AtomicIndex, i:int) -> float:
        j = ind.sitelabel
        dsdir = dot(ind.bravis, privec) + sitedir[j] - sitedir[i]
        return -vddi[ind]*(conjugate(hole.bloch_fun[i])*hole.bloch_fun[i]*conjugate(excit.bloch_fun[j])*excit.bloch_fun[j] - 
                           conjugate(hole.bloch_fun[j])*hole.bloch_fun[i]*conjugate(excit.bloch_fun[i])*excit.bloch_fun[j]*
                           exp(1j*dot(dsdir, dot(inner_dsdir_kfrac, excit.kvec)) - 1j*dot(dsdir, dot(inner_dsdir_kfrac, hole.kvec))))
    return np.real(sum(ele_hole_int_partial(excit, hole, vdd[i], ind, i) for i in range(cell.num_sites) for ind in vdd[i]))

def hartree_fock_solver(sps0:spcl.SingleParticleSystem, vdd:interaction.density_density_type,
                        filling:np.float128, kgrid:Kgrid, controller: Controller, seed, noise = 0.0,
                        save_den_plots=False, save_output=False, saving_dir='/', output_comment='\n',
                        den_plots_suffix = '', output_suffix = '') -> None:
    
    privec = sps0.cell.prim_vecs_dir
    dirtocar = sps0.cell.dirtocar
    num_sites = sps0.cell.num_sites
    sitedir = sps0.cell.sitedir
    hop0_norm = hop_funs.hop_norm(sps0.hoppings)

    ktocar = spcl.fracktocar(dirtocar, privec)
    inner_dsdir_kfrac = dot(transpose(dirtocar), ktocar)
    ite_max = controller.ite_max
    cvg_crit = controller.converge_crit
    mixing = controller.mixing
    cvg_list = []

    kpts = [arr([i/kgrid.enlargement[0]+kgrid.offset[0], j/kgrid.enlargement[1]+kgrid.offset[1]]) 
             for i in range(kgrid.enlargement[0]) 
             for j in range(kgrid.enlargement[1])]
    num_kpts = len(kpts)
    num_ele = round(num_sites*num_kpts*filling)

    eigen_states:List[spcl.EigenState] = []#forward declaration of the variable
    def generate_fock_term(occ_states:List[spcl.EigenState], vddi:Dict[AtomicIndex, np.complex128], i:int, ind: AtomicIndex):
        j = ind.sitelabel
        dsdir = dot(ind.bravis, privec) + sitedir[j] - sitedir[i]
        return -sum(vddi[ind]*np.conjugate(occ.bloch_fun[i])*occ.bloch_fun[j]*\
                    exp(1j*dot(dsdir, dot(inner_dsdir_kfrac, occ.kvec))) for occ in occ_states)
    
    for ite_cycle in range(ite_max):
        if ite_cycle == 0:
            sigma_old = self_energy_init(sps0,vdd,filling,kgrid,seed,noise)
            sigma_tem = hop_funs.hop_copy(sigma_old)
        else:
            den = [sum(np.conjugate(occ.bloch_fun[i])*occ.bloch_fun[i] 
                       for occ in occ_states) for i in range(num_sites)]
            #print("spin up number = ", sum(den[:num_sites//2])) #db
            sigma_tem = [dict() for i in range(num_sites)]
            for i in range(num_sites):
                vddi = vdd[i]
                hop_funs.hop_add_i(sigma_tem, i,
                                   {AtomicIndex(i,(0,0)): sum(vddi[ind]*den[ind.sitelabel] for ind in vddi)},
                                    "inplace") # hartree term
                hop_funs.hop_add_i(sigma_tem, i,
                                   {ind: generate_fock_term(occ_states, vddi, i, ind) for ind in vddi},
                                   "inplace") # fock terms
        sigma_new = hop_funs.hop_add(hop_funs.hop_mul(sigma_tem, mixing), hop_funs.hop_mul(sigma_old, 1-mixing))
        sps = spcl.sps_add_hop(sps0, sigma_new)
        cvg = np.real(hop_funs.hop_norm(hop_funs.hop_add(sigma_new, hop_funs.hop_mul(sigma_old, -1)))/hop0_norm)
        eigen_states = spcl.eigstate_flatten_sort(sps, kpts)
        for eig in eigen_states:
            eig.bloch_fun_renormalize(1/np.sqrt(num_kpts)) #renormalize the bloch function for enlarged PBC
        occ_states = eigen_states[:num_ele] #occupied eigenstates
        free_energy = np.real(1/2*sum(occ.energy for occ in occ_states) + 
                              1/2*num_kpts*sum(dot(np.conjugate(occ.bloch_fun), dot(spcl.hamiltonian(sps0, occ.kvec), occ.bloch_fun)) for occ in occ_states))
        sigma_old = sigma_new
        cvg_list.append((float(cvg), float(free_energy/num_ele)))
        print(ite_cycle, " ", cvg_list[-1]) #db
        if ite_cycle > 0 and cvg < cvg_crit:
            break
    #print(plot_functions.vector_format(np.real(den))) #db
    #plot_functions.den_plot(sps0.cell, eigen_states[num_ele-1].bloch_fun_density()[:num_sites//2]) #db
    #plot_functions.den_plot(sps0.cell, eigen_states[num_ele].bloch_fun_density()[:num_sites//2]) #db
    num_excit_written = 10
    gap_corr_val = tuple((gap_correction(sps0.cell, vdd, eigen_states[num_ele+i], eigen_states[num_ele-1]) for i in range(num_excit_written)))
    
    if save_den_plots:
        plot_functions.spin_plot(sps0.cell, occ_states, savefig=True, showfig=False, 
                                 save_format = os.path.join(saving_dir, 'cs_den_plot'+den_plots_suffix+'.png'))
    if save_output:
        filename = os.path.join(saving_dir, 'output'+output_suffix)
        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                file.write('\n')
        with open(filename, 'a') as file:
            file.write(output_comment)
            file.write(f"convergence = ( {ite_cycle}, {cvg:.4f})\n")
            file.write(f"energy_per_electron = {free_energy/num_ele:.4f}\n")
            file.write(f"mean_field_gap = " 
                       f"{io_fun.tuple_format((eigen_states[num_ele+i].energy - eigen_states[num_ele-1].energy for i in range(num_excit_written)))}\n")
            file.write(f"corrected_gap = "
                       f"{io_fun.tuple_format((eigen_states[num_ele+i].energy - eigen_states[num_ele-1].energy + gap_corr_val[i] for i in range(num_excit_written)))}\n")
            cs_den_total = plot_functions.cs_den_total(sps0.cell, occ_states)
            file.write(f"{privec[0][0]}*{privec[1][1]}_supercell(charge = {cs_den_total[0]:.1f}):\n")
            file.write(f"sx = {cs_den_total[1]:.1f}, "
                       f"sy = {cs_den_total[2]:.1f}, sz = {cs_den_total[3]:.1f}\n")
            sz_excit = (sum(abs(eigen_states[num_ele+j].bloch_fun[i])**2 for i in range(num_sites//2))*num_kpts -\
                  sum(abs(eigen_states[num_ele+j].bloch_fun[i])**2 for i in range(num_sites//2, num_sites))*num_kpts for j in range(num_excit_written))
            file.write(f"sz of the excited states = {io_fun.tuple_format(sz_excit)}\n")
            file.write("\n")
    return sigma_new
    #return ((ite_cycle, cvg), free_energy/num_ele,
    #            eigen_states[num_ele].energy - eigen_states[num_ele-1].energy)
    #eigen_states = spcl.eigstate_flatten_sort(sps, kpts)


