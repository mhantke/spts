#/usr/bin/env python
import argparse
import numpy
import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Particle detection from single particle optical scattering data')
    parser.add_argument('-n','--n-pixels', type=int, help='number of pixels to be masked out from the top', default=1)
    args = parser.parse_args()

    M = numpy.ones(shape=(1024, 2024), dtype=numpy.bool)
    M[:args.n_pixels,:] = False
    
    with h5py.File("mask.h5", "w") as f:
        f["/data"] = M
    
        
