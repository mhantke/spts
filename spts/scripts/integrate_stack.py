#!/usr/bin/env python
import h5py, numpy
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Integrate stack of 2D arrays saved in h5 dataset')
    parser.add_argument('filename', type=str, help='filename')
    parser.add_argument('dataset', type=str, help='dataset')
    args = parser.parse_args()

    with h5py.File(args.filename, "r") as f:
        print(f.keys())
        ds = f[args.dataset]
        N = ds.shape[0]
        sum = numpy.zeros(shape=(ds.shape[1],ds.shape[2]))
        for i in range(N):
            print("(%i/%i)" % (i+1,N))
            sum += ds[i,:,:]

    with h5py.File("sum.h5", "w") as f:
        f["data"] = sum
    
        
        
