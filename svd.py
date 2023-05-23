import scipy
import numpy as np

A = np.load("training_matrix.npy");

print("Loaded!");

u, s, vh = scipy.linalg.svd(A, full_matrices=True,compute_uv=True,overwrite_a=True);

print("SVD complete!");

np.save("u_matrix", u);
np.save("s_matrix", s);
np.save("vh_matrix", vh);
