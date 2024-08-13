h0 = 13.54295
h_ms = h0*(0.37222232-0.25)
h_mm = h0*0.5
h = 50.0
dis_upper_dir = (0.0, 0.0)
with open('./DFT_requirement/POSCAR', 'w') as file:
    file.write("Mo2 Se4\n")
    file.write("1.0\n")
    file.write("1.6611300941223299   -2.8771617210015465    0.0000000000000000\n")
    file.write("1.6611300941223299    2.8771617210015465    0.0000000000000000\n")
    file.write(f"0.0000000000000000    0.0000000000000000    {h:.15f}\n")
    file.write("Mo Se\n")
    file.write("2 4\n")
    file.write("direct\n")
    file.write(f"{1/3:.16f}    {2/3:.16f}    {0.5 + (-h_mm/2)/h:.16f} Mo4+\n")
    file.write(f"{1/3 + dis_upper_dir[0]:.16f}    {2/3 + dis_upper_dir[1]:.16f}    {0.5 + (h_mm/2)/h:.16f} Mo4+\n")
    file.write(f"{2/3:.16f}    {1/3:.16f}    {0.5 + (-h_mm/2 - h_ms)/h:.16f} Se2-\n")
    file.write(f"{2/3:.16f}    {1/3:.16f}    {0.5 + (-h_mm/2 + h_ms)/h:.16f} Se2-\n")
    file.write(f"{2/3 + dis_upper_dir[0]:.16f}    {1/3 + dis_upper_dir[1]:.16f}    {0.5 + (h_mm/2 - h_ms)/h:.16f} Se2-\n")
    file.write(f"{2/3 + dis_upper_dir[0]:.16f}    {1/3 + dis_upper_dir[1]:.16f}    {0.5 + (h_mm/2 + h_ms)/h:.16f} Se2-\n")