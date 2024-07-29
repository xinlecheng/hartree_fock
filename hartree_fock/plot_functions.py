import matplotlib.pyplot as plt
import numpy as np
import manipulation_functions_for_hoppings as hop_funs
import single_particle_class as spcl
from typing import List
def dot(*arg):
    return np.matmul(*arg)

def list_plot(datasets):
    plt.figure()
    for data in datasets:
        x, y  = zip(*data)
        plt.plot(x, y, marker='o')
    plt.show()
def den_plot(cell: spcl.Cell, den):
    a_m = np.linalg.norm(cell.dirtocar[:,0])
    radius = 0.25*a_m
    fig, ax = plt.subplots()
    den = np.real(den)
    den = den/np.max(den)
    for i in range(len(den)):
        position = np.linalg.matmul(cell.dirtocar, cell.sitedir[i])
        color = (0.95, 0.95*(1-den[i]), 0.95*(1-den[i]))
        circle = plt.Circle(tuple(position), radius, color=color)
        ax.add_artist(circle)
    a1 = np.linalg.matmul(cell.dirtocar, cell.prim_vecs_dir[0])
    a2 = np.linalg.matmul(cell.dirtocar, cell.prim_vecs_dir[1])
    x_min = np.min([0, a1[0], a2[0], a1[0]+a2[0]])
    x_max = np.max([0, a1[0], a2[0], a1[0]+a2[0]])
    y_min = np.min([0, a1[1], a2[1], a1[1]+a2[1]])
    y_max = np.max([0, a1[1], a2[1], a1[1]+a2[1]])
    ax.set_xlim(x_min-radius, x_max+radius)
    ax.set_ylim(y_min-radius, y_max+radius)
    ax.set_aspect("equal")
    plt.show()

def den_subplot(ax:plt.Axes, cell:spcl.Cell, den, title=''):
    """
    does inplace change to ax, assume that density is real but not necessary positive
    """
    a_m = np.linalg.norm(cell.dirtocar[:,0])
    radius = 0.25*a_m
    for i in range(len(den)):
        position = np.linalg.matmul(cell.dirtocar, cell.sitedir[i])
        if den[i] >= 0:
            color = (0.95, 0.95*(1-den[i]), 0.95*(1-den[i]))
        else:
            color = (0.95*(1+den[i]), 0.95*(1+den[i]), 0.95)
        circle = plt.Circle(tuple(position), radius, color=color)
        ax.add_artist(circle)
    a1 = np.linalg.matmul(cell.dirtocar, cell.prim_vecs_dir[0])
    a2 = np.linalg.matmul(cell.dirtocar, cell.prim_vecs_dir[1])
    x_min = np.min([0, a1[0], a2[0], a1[0]+a2[0]])
    x_max = np.max([0, a1[0], a2[0], a1[0]+a2[0]])
    y_min = np.min([0, a1[1], a2[1], a1[1]+a2[1]])
    y_max = np.max([0, a1[1], a2[1], a1[1]+a2[1]])
    ax.set_xlim(x_min-radius, x_max+radius)
    ax.set_ylim(y_min-radius, y_max+radius)
    ax.set_aspect("equal")
    ax.set_title(title)

def spin_plot(cell:spcl.Cell, occ_states:List[spcl.EigenState], showfig=False, savefig=False, 
              save_format='tem.png', title=''):
    s0 = np.array([[1,0],[0,1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    sv = [s0, sx, sy, sz]
    num_sites = cell.num_sites//2
    def calculate_cs_den(bloch_fun:np.ndarray, i:int):
        spinor_i = np.array([bloch_fun[i], bloch_fun[i+num_sites]])
        return np.array([np.real(dot(np.conjugate(spinor_i), dot(s, spinor_i))) for s in sv])
    cs_den = [sum(calculate_cs_den(occ.bloch_fun, i) for occ in occ_states) for i in range(cell.num_sites//2)]
    c_den, sx_den, sy_den, sz_den = zip(*cs_den)
    c_den_max = np.max(c_den)
    c_den = c_den/c_den_max
    sx_den = sx_den/c_den_max
    sy_den = sy_den/c_den_max
    sz_den = sz_den/c_den_max
    fig, axs = plt.subplots(2,2)
    den_subplot(axs[0,0], cell, c_den, title="charge")
    den_subplot(axs[0,1], cell, sz_den, title="spin_z")
    den_subplot(axs[1,0], cell, sx_den, title="spin_x")
    den_subplot(axs[1,1], cell, sy_den, title="spin_y")
    fig.suptitle(title)
    if savefig:
        fig.savefig(save_format)
    if showfig:
        plt.show()


def element_format(x):
    return f"{x:.2f}"
vector_format = np.vectorize(element_format)

if __name__ == "__main__":
    plt.Circle((1.0,1.0),1.0, color=(1,0,0))