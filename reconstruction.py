import numpy as np
import json
from PIL import Image, ImageOps

s = np.load("s_matrix.npy");
u = np.load("u_matrix.npy");
vh = np.load("vh_matrix.npy");

sL = [];
uL = [];
face_components = {};

training_data = {};
with open("training_data_specs.json", "r") as f:
    training_data = json.load(f);

d = training_data["training_images"];

K = 50; # components in a face

face_bases = [:, 0:K];
s_vals = [:,0:K];

c_img = d[0];
print("Checking image {}".format(c_img));


