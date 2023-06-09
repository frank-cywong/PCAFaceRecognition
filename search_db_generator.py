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

K = 100; # components in a face

face_bases = u[:, 0:K];
s_vals = s[0:K];

#face_one = d[0];
#coeffs_one = [0, 0:K];

#c_matrix = np.zeros(len(face_bases),dtype=np.float32);

#for i in range(K):
#    c_matrix += s[i] * vh[i][0] * u[:,i];

dir_string = "../img_align_celeba/"

index = 0;
total = len(d);
AA = [];

for fn in d:
    if(index % 100 == 0):
        print("Currently at {} out of {}".format(index, total));
    im = Image.open(dir_string + fn);
    im2 = im.resize((89, 109));
    im2 = ImageOps.grayscale(im2);

    l = list(Image.Image.getdata(im2));
    A = np.array(l, dtype=np.float32)
    A /= 255.0;

    average_face = np.load("average_face.npy");

    A -= average_face;

    #new_matrix_build = np.zeros(len(A), dtype=np.float32);

    coeffs = [];

    for i in range(K):
        coeffs.append(float(np.dot(A, face_bases[:,i])));
    
    index += 1;
    AA.append([coeffs, fn]);

g = open("reduced_images.json", "w");
json.dump(AA, g);
