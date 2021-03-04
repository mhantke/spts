#!/usr/bin/env python
import numpy as np
import argparse
import os, sys, shutil
import time

import argparse

parser = argparse.ArgumentParser(description='Mie scattering imaging data analysis')
parser.add_argument('mode', metavar='M', type=str, nargs=1,
                    help='execution mode can be single, mulpro, or mpi')
parser.add_argument('-v', '--verbose', dest='verbose',  action='store_true', help='verbose mode', default=False)
parser.add_argument('-d', '--debug', dest='debug',  action='store_true', help='debugging mode (even more output than in verbose mode)', default=False)
args = parser.parse_args()

mode = args.mode[0]

curdir = os.path.abspath(".")
files = os.listdir(".")

# Check argument
if mode == "single":
    ntasks = 1
    cmd = "run_spts.py"
elif mode == "mulpro":
    ntasks = 12
    cmd = ("run_spts.py -c %i" % ntasks)    
elif mode == "mpi":
    ntasks = 12
    cmd = "mpirun run_spts.py -m"
else:
    print("ERROR: The selected mode argument (%s) is invalid. It must be either single, mulpro, or mpi." % mode)
    sys.exit(1)

if args.debug:
    cmd += " -d"
elif args.verbose:
    cmd += " -v"

# Check whether spts.conf exists
if "spts.conf" not in files:
    print("ERROR: spts.conf must exist in current directory.")
    sys.exit(1)

# Check whether at least one CXI file exists
cxi_filenames = [f for f in files if f.endswith(".cxi")]
if len(cxi_filenames) == 0:
    print("WARNING: At least one CXI file must exist in current directory. Otherwise there is nothing to do.")
    sys.exit(1)

# Replace line with filename for making spts.conf universal
with open("%s/spts.conf" % curdir, "r") as f:
    spts_conf_lines = []
    for line in f.readlines():
        if line.startswith("filename"):
            spts_conf_lines.append("filename = ./frames.cxi\n")
        else:
            spts_conf_lines.append(line)

for fn in cxi_filenames:
    n = fn[:-4]    

    # Create output directory if it does not exist
    d = "./"+n+"_analysis"
    if os.path.exists(d):
        print("WARNING: Directory %s for output already exists." % d)
        #shutil.rmtree(d)
    else:
        os.mkdir(d)

    # Write link to data file
    os.system("ln -sf ../%s %s/frames.cxi\n" % (fn, d))
        
    # Write universal spts.conf
    with open("%s/spts.conf" % d, "w") as f:
        f.writelines(spts_conf_lines)
        
    # Write bash submission script
    lines = ["#!/bin/bash\n",
             "sbatch --output=spts.out --job-name=spts ./spts.sh\n"]
    sc_sub = ("%s/submit_spts" % d)
    with open(sc_sub, "w") as f:
        f.writelines(lines)
        
    # Write slurm script
    lines = ["#!/bin/sh\n",
             "#SBATCH --nodes=1-1\n",
             "#SBATCH --ntasks=%i\n" % ntasks,
             "#SBATCH --cpus-per-task=1\n",
             "#SBATCH --partition=fast\n",
             "#SBATCH --exclude=c018\n"
             "%s\n" % cmd]
    sc_slu = ("%s/spts.sh" % d)
    with open(sc_slu, "w") as f:
        f.writelines(lines)  

    # Make submission script executable
    os.system("chmod ug+x %s" % sc_sub)
    # Go into output directory and fire up submission script 
    os.system("cd %s; ./submit_spts" % (d))

