import numpy as np
import json
from PIL import Image, ImageOps

s = np.load("s_matrix.npy");
u = np.load("u_matrix.npy");
vh = np.load("vh_matrix.npy");

print("Loaded");

sp = np.zeros((len(u), len(vh)), dtype=np.float32);
for i in range(len(s)):
    sp[i][i] = s[i];

print("Multiplying");

A = sp @ vh;

print("Multiplied");

B = u @ A;

print("Multiplied part 2");

O = np.load("training_matrix_v2.npy");

print("Loaded O");

E = O - B;

np.save("error_matrix", E);

print(E);
