import json

mapobj = {};

with open("identity_CelebA.txt", "r") as f:
    l = ""
    l = f.readline()
    while(l):
        s = l.split(" ");
        d = int(s[1]);
        img = s[0];
        #print(img);
        #print(d);
        if not d in mapobj:
            mapobj[d] = [];
        mapobj[d].append(img);
        l = f.readline()

f = open("celeb_mappings.json", "w");
json.dump(mapobj, f);