import numpy as np;
from PIL import Image, ImageOps;

def array_to_image(a):
    a = a.reshape(109, 89);
    a *= 255.0;
    # This converts the entires of the data matrix into a form recognizable by the image object 
    Approx = a.astype(np.uint8)
    # Creates a new image from the data array
    im3 = Image.fromarray(Approx,'L')
    im3.show();

if __name__ == "__main__":
    average_face = np.load("average_face.npy");
    array_to_image(average_face);