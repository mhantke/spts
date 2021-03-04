#!/usr/bin/env python
import numpy as np
import argparse
import os, sys, shutil
import time

filenames = [f for f in os.listdir(".") if f.endswith(".cxi")]

if len(sys.argv) > 1:
    timestamp = sys.argv[1]
    filenames = [filename for filename in filenames if timestamp in filename]

for f in filenames:
    n = f[:-4]
    
    d = "./"+n+"_analysis"
    if not os.path.exists(d):
        os.mkdir(d)

    cmds = ["cp ./spts.conf %s/" % (d), "ln -s ../%s %s/frames.cxi" % (f,d), "cd %s; run_spts.py -c 9 -v; cd .." % (d)]
    for cmd in cmds:
        print(cmd) 
        os.system(cmd)
    
