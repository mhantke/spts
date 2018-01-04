#!/bin/env python
import os,sys

curdir = os.path.abspath(".")

for f in [f for f in os.listdir(curdir) if f.endswith(".avi")]:
    fout = f[:-4] + ".out"
    cmd = "sbatch --output=%s/%s avi_to_h5.sh %s/%s" % (curdir, fout, curdir, f)
    for arg in sys.argv[1:]:
        cmd += " " + arg
    print cmd
    os.system(cmd)
