import numpy as np
import plot_functions
import single_particle_class

# region Custom Region
def arr(*arg):
    return np.array(*arg)
def dot(*arg):
    return np.matmul(*arg)
pi = np.pi
def inv(*arg):
    return np.linalg.inv(*arg)
def norm(*arg):
    return np.linalg.norm(*arg)
def conjugate(*arg):
    return np.conjugate(*arg)
def transpose(*arg):
    return np.transpose(*arg)
def exp(arg):
    return np.exp(arg)
def sqrt(arg):
    return np.sqrt(arg)
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
def kronecker_delta(i,j):
    if i==j:
        return 1
    else:
        return 0
def functional_if(crit, true_return, false_return=None):
    if crit:
        return true_return
    else:
        return false_return
#endregion

def create_k_lattice_dir(l:int):
    kl_single_layer = [arr([i,j]) for i in range(-l, l+1) for j in range(-l, l+1) if abs(i+j) <= l]
    return [('t', kl_single_layer[i]) for i in range(len(kl_single_layer))] + \
            [('b', kl_single_layer[i]) for i in range(len(kl_single_layer))]

def create_kspace_hamiltonian(k:np.ndarray, kl:list, theta=0, m_eff=0, moire_v=0, phi=0, tunnel_w=0, delta=0):
    """
    takes k in cartesian coordinates
    m_eff is the effective electron/hole mass
    moire_v controls the strength of moire potential
    phi controls the landscape of moire potential
    tunnel_w is the interlayer tunneling strength
    delta is the interlayer displacement field
    """
    num_sites = len(kl)
    g1 = arr([1, 0])*4*pi*theta/sqrt(3)
    g2 = arr([1/2, sqrt(3)/2])*4*pi*theta/sqrt(3)
    kp = 1/3*(-2*g1 + g2)
    km = 1/3*(-g1-g2)
    tol = 10**(-8) #numerical tolerance
    nearest_recipo_vecs = [g1, g2, -g1+g2, -g1, -g2, g1-g2, arr([0,0])] #recipocal vecs in cartesian coordinates
    
    hamiltonian = np.zeros((num_sites, num_sites), dtype=np.complex128)
    for i in range(num_sites):
        for j in range(num_sites):
            rlt_kvec = kl[i][1]-kl[j][1] #rlt for 'relative'
            if norm(rlt_kvec) > norm(g1) + tol:
                continue
            isvec = [np.allclose(rlt_kvec, recipo_vec) for recipo_vec in nearest_recipo_vecs]
            if kl[i][0] == 't' and kl[j][0] == 't':
                hamiltonian[i,j] += kronecker_delta(i,j)*1/2/m_eff*norm(kl[i][1] + k - kp)**2 +\
                functional_if(isvec[0] or isvec[2] or isvec[4], moire_v*exp(1j*phi), 0) +\
                functional_if(isvec[1] or isvec[3] or isvec[5], moire_v*exp(-1j*phi), 0) + delta/2
            if kl[i][0] == 'b' and kl[j][0] == 'b':
                hamiltonian[i,j] += kronecker_delta(i,j)*1/2/m_eff*norm(kl[i][1] + k - km)**2 +\
                functional_if(isvec[0] or isvec[2] or isvec[4], moire_v*exp(-1j*phi), 0) +\
                functional_if(isvec[1] or isvec[3] or isvec[5], moire_v*exp(1j*phi), 0) - delta/2
            if kl[i][0] == 't' and kl[j][0] == 'b':
                hamiltonian[i,j] += functional_if(isvec[1] or isvec[2] or isvec[6], tunnel_w, 0)
            if kl[i][0] == 'b' and kl[j][0] == 't':
                hamiltonian[i,j] += functional_if(isvec[4] or isvec[5] or isvec[6], tunnel_w, 0)
    return hamiltonian

def bandstructure(hamiltonian, ktocar, kpath: list):
    '''
    output the bandstructure of a certain hamiltonian function along kpath
    assume that hamiltonian takes k in cartesian coordinates
    '''
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
        v = eigvalsh(hamiltonian(dot(ktocar, kpoints[i][1])))
        tem.append([(kpoints[i][0], v[j]) for j in range(len(v))])
    return transpose(tem,(1,0,2))  

if __name__ == '__main__':
    
    kl_dir = create_k_lattice_dir(5)
    theta = 4.0*pi/180
    g1 = arr([1, 0])*4*pi*theta/sqrt(3)
    g2 = arr([1/2, sqrt(3)/2])*4*pi*theta/sqrt(3)
    frac_to_car = transpose([g1, g2])
    kl = [(element[0], dot(frac_to_car, element[1])) for element in kl_dir]
    def hamiltonian(k):
        return create_kspace_hamiltonian(k, kl, theta, -0.43*1.5*10**(-3), 15, 140*pi/180, -13, 0)
    kline = [arr([0,0]), 20, arr([1/3,1/3]), 10, arr([1/2,0]), 20, arr([0,0])]
    bs = bandstructure(hamiltonian, frac_to_car, kline)
    print("bs calcualted!")
    num_bands = len(bs)
    plot_functions.list_plot(bs[num_bands-8:num_bands], aspect_ratio=1/250)
    #create_kspace_hamiltonian((0,0), kl)
    
    