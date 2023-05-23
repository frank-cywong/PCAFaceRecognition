import numpy as np

s = np.load("s_matrix.npy");

total_variance = np.sum(s);
print("Total variance: {}".format(total_variance));

print("Total vector count: {}".format(len(s)));

c_var = 0;
c_cnt = 0;

cutoff50 = total_variance * 0.5;
cutoff55 = total_variance * 0.55;
cutoff60 = total_variance * 0.6;
cutoff65 = total_variance * 0.65;
cutoff70 = total_variance * 0.7;
cutoff75 = total_variance * 0.75;
cutoff80 = total_variance * 0.8;
cutoff85 = total_variance * 0.85;
cutoff90 = total_variance * 0.9;
cutoff95 = total_variance * 0.95;

for v in s:
    c_var += v;
    c_cnt += 1;
    if(c_cnt < 15):
        print("Variance of vector {}: {}".format(c_cnt, v));
    if(c_var > cutoff50):
        print("Using the first {} vectors are needed to get 50% of the variance".format(c_cnt));
        cutoff50 = total_variance * 1.1; # hacky but works
    if(c_var > cutoff55):
        print("Using the first {} vectors are needed to get 55% of the variance".format(c_cnt));
        cutoff55 = total_variance * 1.1; # hacky but works
    if(c_var > cutoff60):
        print("Using the first {} vectors are needed to get 60% of the variance".format(c_cnt));
        cutoff60 = total_variance * 1.1; # hacky but works
    if(c_var > cutoff65):
        print("Using the first {} vectors are needed to get 65% of the variance".format(c_cnt));
        cutoff65 = total_variance * 1.1; # hacky but works
    if(c_var > cutoff70):
        print("Using the first {} vectors are needed to get 70% of the variance".format(c_cnt));
        cutoff70 = total_variance * 1.1; # hacky but works
    if(c_var > cutoff75):
        print("Using the first {} vectors are needed to get 75% of the variance".format(c_cnt));
        cutoff75 = total_variance * 1.1; # hacky but works
    if(c_var > cutoff80):
        print("Using the first {} vectors are needed to get 80% of the variance".format(c_cnt));
        cutoff80 = total_variance * 1.1; # hacky but works
    if(c_var > cutoff85):
        print("Using the first {} vectors are needed to get 85% of the variance".format(c_cnt));
        cutoff85 = total_variance * 1.1; # hacky but works
    if(c_var > cutoff90):
        print("Using the first {} vectors are needed to get 90% of the variance".format(c_cnt));
        cutoff90 = total_variance * 1.1; # hacky but works
    if(c_var > cutoff95):
        print("Using the first {} vectors are needed to get 95% of the variance".format(c_cnt));
        break;