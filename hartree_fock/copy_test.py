import numpy as np
from manipulation_functions_for_hoppings import AtomicIndex
import manipulation_functions_for_hoppings
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hubbard_u', type=float) #0.2
#parser.add_argument('cutoff', type=int) #6
parser.add_argument('--scaling', type=float) #436 for epsilon=10
args = parser.parse_args()
saving_dir = './results'
filename = os.path.join(saving_dir, 'test_output')
if not os.path.exists(filename):
    with open(filename, 'w') as file:
        file.write(args.hubbard_u, " ", args.scaling)