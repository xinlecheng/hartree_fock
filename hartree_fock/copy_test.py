import numpy as np
from manipulation_functions_for_hoppings import AtomicIndex
import manipulation_functions_for_hoppings
import time

def f(x,y,z=0,k=1):
    return x+y+z+k
tu1 = (100,(101,102))
tu2 = (100,tuple(np.array([101,102])))
# tu1 = 102019201
# tu2 = np.int64(102019201)
# print("tu1 = ", tu1)
# print("tu2 = ", tu2)
# print("hash(tu1) = ", hash(tu1))
# print("hash(tu2) = ", hash(tu2))
# print("id(tu1) = ", id(tu1))
# print("id(tu2) = ", id(tu2))
# print("tu1 == tu2 ? ", tu1 == tu2)
# print([i+j for i in range(2) for j in range(2)])
# print(type(np.int32(1)+round(np.float128(0.9))))
# print(np.round(1.9999999))
ori = [0,1,2,3]
# isinstance(1, manipulation_functions_for_hoppings.Hoppings)
# print(len(np.array([[1,1],[1,1],[1,1]])))
# print(np.array([1,1])==np.array([1,1]))
dataset = [[1,2],[3,4],[5,6]]
print(round(time.time()))
time.sleep(1)
print(round(time.time()))
#print(type(np.mod((1,1),(1,1))))


import os
#os.path.join(a, b)