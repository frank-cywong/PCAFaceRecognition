import numpy as np
import json
from PIL import Image, ImageOps
from array_to_image import array_to_image

#s = np.load("s_matrix.npy");
u = np.load("u_matrix.npy");
#vh = np.load("vh_matrix.npy");

sL = [];
uL = [];
face_components = {};

#training_data = {};
#with open("training_data_specs.json", "r") as f:
#    training_data = json.load(f);

#d = training_data["training_images"];

print("Loaded data!");

K = 600; # components in a face

face_bases = u[:, 0:K];
#s_vals = s[0:K];

#face_one = d[0];
#coeffs_one = [0, 0:K];

#c_matrix = np.zeros(len(face_bases),dtype=np.float32);

#for i in range(K):
#    c_matrix += s[i] * vh[i][0] * u[:,i];

dir_string = "../img_align_celeba/"

target_one = "000002.jpg"
target_two = "000007.jpg"

im = Image.open(dir_string + target_one);
im2a = im.resize((89, 109));
im2a = ImageOps.grayscale(im2a);

l = list(Image.Image.getdata(im2a));
A = np.array(l, dtype=np.float32)
A /= 255.0;

average_face = np.load("average_face.npy");

A -= average_face;

new_matrix_build = np.zeros(len(A), dtype=np.float32);

coeffs = [];

for i in range(K):
    coeffs.append(np.dot(A, face_bases[:,i]));

im = Image.open(dir_string + target_two);
im2b = im.resize((89, 109));
im2b = ImageOps.grayscale(im2b);

l = list(Image.Image.getdata(im2b));
A = np.array(l, dtype=np.float32)
A /= 255.0;

A -= average_face;

new_matrix_build = np.zeros(len(A), dtype=np.float32);

coeffs2 = [];

for i in range(K):
    coeffs2.append(np.dot(A, face_bases[:,i]));

# merging

for i in range(K):
    new_matrix_build += face_bases[:,i] * (coeffs[i] + coeffs2[i]) * 0.5;

#print(coeffs);

new_matrix_build += average_face;

new_matrix_build = np.clip(new_matrix_build, 0.0, 1.0);

print("Encoded:");
array_to_image(new_matrix_build);
print("Actual 1:");
im2a.show();
print("Actual 2:");
im2b.show();