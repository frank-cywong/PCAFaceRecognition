import numpy as np
import json
from PIL import Image, ImageOps
from array_to_image import array_to_image

s = np.load("s_matrix.npy");
u = np.load("u_matrix.npy");
#vh = np.load("vh_matrix.npy");

sL = [];
uL = [];
face_components = {};

training_data = {};
with open("training_data_specs.json", "r") as f:
    training_data = json.load(f);

d = training_data["training_images"];

print("Loaded data!");

K = 400; # components in a face

face_bases = u[:, 0:K];
s_vals = s[0:K];

#face_one = d[0];
#coeffs_one = [0, 0:K];

#c_matrix = np.zeros(len(face_bases),dtype=np.float32);

#for i in range(K):
#    c_matrix += s[i] * vh[i][0] * u[:,i];

dir_string = "../img_align_celeba/"

im = Image.open(dir_string + d[0]);
im2 = im.resize((89, 109));
im2 = ImageOps.grayscale(im2);

l = list(Image.Image.getdata(im2));
A = np.array(l, dtype=np.float32)
A /= 255.0;

average_face = np.load("average_face.npy");

A -= average_face;

new_matrix_build = np.zeros(len(A), dtype=np.float32);

coeffs = [];

for i in range(K):
    coeffs.append(np.dot(A, face_bases[:,i]));

for i in range(K):
    new_matrix_build += face_bases[:,i] * coeffs[i];

#print(coeffs);

new_matrix_build += average_face;

new_matrix_build = np.clip(new_matrix_build, 0.0, 1.0);

print("Encoded:");
array_to_image(new_matrix_build);
print("Actual:");
im2.show();