import numpy as np
h_ni = 1.55812
h = 25.0
strain = 0.0
n1 = 5
n2 = 3
with open('./POSCAR', 'w') as file:
    file.write(f"Ni{n1*n2} I{2*n1*n2}\n")
    file.write("1.0\n")
    file.write(f"{(1 + strain)*1.975*n1:.15f}    {-np.sqrt(3)*(1 + strain)*1.975*n1:.15f}    {0.0:.15f}\n")
    file.write(f"{(1 + strain)*1.975*n2:.15f}    {np.sqrt(3)*(1 + strain)*1.975*n2:.15f}    {0.0:.15f}\n")
    file.write(f"{0.0:.15f}    {0.0:.15f}    {h:.15f}\n")
    file.write("Ni I\n")
    file.write(f"{n1*n2} {2*n1*n2}\n")
    file.write("direct\n")
    for i2 in range(n2):
        for i1 in range(n1):
            file.write(f"{(0.0 + i1)/n1:.16f}    {(0.0 + i2)/n2:.16f}    {0.5:.16f}\n")
    for i2 in range(n2):
        for i1 in range(n1):
            file.write(f"{(2/3 + i1)/n1:.16f}    {(1/3 + i2)/n2:.16f}    {0.5 - h_ni/h :.16f}\n")
    for i2 in range(n2):
        for i1 in range(n1):
            file.write(f"{(1/3 + i1)/n1:.16f}    {(2/3 + i2)/n2:.16f}    {0.5 + h_ni/h:.16f}\n")