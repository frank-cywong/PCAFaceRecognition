import json

mapobj = {};

with open("celeb_mappings.json", "r") as f:
    mapobj = json.load(f)

ref = {};
used = [];

trainwith = [];

imgcount = 0;
for k in mapobj:
    a = mapobj[k];
    lv = a.pop();
    ref[k] = lv;
    used.append(k);
    for n in a:
        trainwith.append(n);
    imgcount += len(a);
    if(imgcount >= 30000):
        break
unused = [];
for k in mapobj:
    if k not in used:
        unused.append(k);
data = {};
data["reference_images"] = ref;
data["used_identities"] = used;
data["training_images"] = trainwith;
data["unused_identities"] = unused;

g = open("training_data_specs.json", "w");
json.dump(data, g);
