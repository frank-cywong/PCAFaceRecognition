import scipy
import numpy as np

A = np.load("training_matrix_v2.npy");

print(A.dtype);

print("Loaded!");

u, s, vh = scipy.linalg.svd(A, full_matrices=True,compute_uv=True,overwrite_a=True,lapack_driver="gesvd");

print("SVD complete!");

np.save("u_matrix_v2", u);
np.save("s_matrix_v2", s);
np.save("vh_matrix_v2", vh);
