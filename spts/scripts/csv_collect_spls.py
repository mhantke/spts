#!/usr/bin/env python
import os, pickle
import numpy as np

p_files = [f for f in os.listdir("./") if f.startswith("params") and f.endswith(".p")]
p_files.sort()

D = {}

#print len(p_files)

for i,p_file in enumerate(p_files):
     D_i = pickle.load(open(p_file, "r"))
     
     if i == 0:
	  keys = D_i.keys()
	  keys.sort()
          keys.remove("timestamp")
          keys.insert(0, "timestamp")
       	  for k in keys:
               if isinstance(D_i[k], str):
                    D[k] = [""]*len(p_files)
               else:
       	            D[k] = np.zeros(len(p_files), dtype=type(D_i[k]))
               
     for k in keys:
          print p_file, k
    	  D[k][i] = D_i[k]


with open("params_spts.csv", "w") as f:
     l = ""
     for k in keys:
          l += k + "\t"
     l = l[:-1] + "\n"
     f.write(l)
     for i in range(len(p_files)):
          l = ""
          for k in keys:
               l += str(D[k][i]) + "\t"
          l = l[:-1] + "\n"
          f.write(l)
