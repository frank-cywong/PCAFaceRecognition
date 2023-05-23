import json
from PIL import Image, ImageOps
import numpy as np

dir_string = "../img_align_celeba/"

training_data = {};
with open("training_data_specs.json", "r") as f:
    training_data = json.load(f);

d = training_data["training_images"];
index = 0;
total = len(d);
AA = [];
for entry in d:
    if(index % 100 == 0):
        print("Currently at {} out of {}".format(index, total));
    im = Image.open(dir_string + entry);
    im2 = im.resize((89, 109));
    im2 = ImageOps.grayscale(im2);
    #im2.show();
    index += 1
    l = list(Image.Image.getdata(im2));
    #print(l);
    #print(len(l));
    A = np.array(l, dtype=np.float32)
    A /= 255.0;
    avg = np.average(A);
    A -= avg;
    #print(A);
    AA.append(A);
    im.close();
print("Stacking:");
B = np.stack(AA, axis=-1, dtype=np.float32);
print(len(B));
print(len(B[0]));
np.save("training_matrix", B);