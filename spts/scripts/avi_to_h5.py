#!/usr/bin/env python
import numpy as np
import os,sys,h5py
from pylab import imread
import argparse

def avi_to_frames(filename):
    d = filename[:-len(".avi")] + "_frames"
    if os.path.exists(d):
        print("Frames already converted to PNG ...")
        return d
    print("Prepare output directory ...")
    cmd = "mkdir %s" % d
    print(cmd)
    os.system(cmd)
    cmd = "rm %s/*" % d
    print(cmd)
    os.system(cmd)
    print("Convert AVI to PNGs ...")
    cmd = ("ffmpeg -i %s %s/" % (filename,d) + "frame%09d.png")
    print(cmd)
    os.system(cmd)
    return d

def _new_dataset(f, name, dtype, frame_shape, N=None):
    if N is not None:
        shape = tuple([N] + list(frame_shape))
    else:
        shape = frame_shape
    ds = f.create_dataset(name, shape, dtype=dtype)
    ndim = len(list(frame_shape))
    if N is not None:
        axid = "experiment_identifier"
        if ndim == 1:
            axid += ":x"
        elif ndim == 2:
            axid += ":y:x"
        elif ndim == 3:
            axid += ":z:y:x"
    else:
        if ndim == 1:
            axid = "x"
        elif ndim == 2:
            axid = "y:x"
        elif ndim == 3:
            axid = "z:y:x"           
    ds.attrs.modify("axes",[axid])
    return ds
    
def frames_to_h5(directory, factor, N_max=250):
    filenames = [(directory + "/" + filename) for filename in os.listdir(directory) \
         if filename.startswith("frame") and filename.endswith(".png")]
    #filenames = filenames[:10]
    filename_h5 = directory + "/frames.cxi"
    init = False
    N = len(filenames)
    print("Convert PNGs to H5 ...")
    with h5py.File(filename_h5,"w") as f:
        f.create_group("/entry_1")
        # Raw data
        grp_name = "/entry_1/data_1"
        ds_raw_sum = _new_dataset(f, grp_name + "/sum", np.int32, tuple(), N)
        ds_raw_min = _new_dataset(f, grp_name + "/min", np.int16, tuple(), N)
        ds_raw_max = _new_dataset(f, grp_name + "/max", np.int16, tuple(), N)
        ds_raw_median = _new_dataset(f, grp_name + "/median", np.float64, tuple(), N)
        for i,filename in enumerate(filenames):
            print("(%i/%i) Raw frame to H5" % (i+1,N))
            img = imread(filename)
            # RGB to Grayscale
            img = np.array(np.round(255*img.sum(axis=2)/3),dtype=np.int16)
            if not init:
                ds_raw = _new_dataset(f, grp_name + "/data", np.int16, img.shape, N)
                init = True
            ds_raw[i,:,:] = img[:,:]
            ds_raw_sum[i] = img.sum()
            ds_raw_min[i] = img.min()
            ds_raw_max[i] = img.max()
            ds_raw_median[i] = np.median(img)  
        # Background and fluctuation
        ds_std = _new_dataset(f, grp_name + "/std", np.float64, img.shape, None)
        ds_bg = _new_dataset(f, grp_name + "/bg", np.float64, img.shape, None)
        if N < N_max:
            bg = np.median(ds_raw,axis=0)
            std = np.std(ds_raw,axis=0)
        else:
            bg = np.median(ds_raw[:N_max,:,:],axis=0)
            std = np.std(ds_raw[:N_max,:,:],axis=0)
        ds_std[:,:] = std[:,:]
        ds_bg[:,:] = bg[:,:]
        # Corrected data 1
        grp_name = "/entry_1/image_1"
        f.create_group(grp_name)
        ds_corr1_sum = _new_dataset(f, grp_name + "/sum", np.int32, tuple(), N)
        ds_corr1_min = _new_dataset(f, grp_name + "/min", np.int16, tuple(), N)
        ds_corr1_max = _new_dataset(f, grp_name + "/max", np.int16, tuple(), N)
        ds_corr1_median = _new_dataset(f, grp_name + "/median", np.float64, tuple(), N)
        ds_corr1 = _new_dataset(f, grp_name + "/data", np.float64, img.shape, N)
        for i in range(N):
            print("(%i/%i) Corrected (1) frame to H5" % (i+1,N))
            temp = ds_raw[i,:,:] - bg[:,:]
            img1 = temp - np.median(temp)
            ds_corr1[i,:,:] = img1[:,:]
            ds_corr1_sum[i] = img1.sum()
            ds_corr1_min[i] = img1.min()
            ds_corr1_max[i] = img1.max()
            ds_corr1_median[i] = np.median(img1)
        # Corrected data 1
        grp_name = "/entry_1/image_2"
        f.create_group(grp_name)
        ds_corr2_sum = _new_dataset(f, grp_name + "/sum", np.int32, tuple(), N)
        ds_corr2_min = _new_dataset(f, grp_name + "/min", np.int16, tuple(), N)
        ds_corr2_max = _new_dataset(f, grp_name + "/max", np.int16, tuple(), N)
        ds_corr2_median = _new_dataset(f, grp_name + "/median", np.float64, tuple(), N)
        ds_corr2 = _new_dataset(f, grp_name + "/data", np.float64, img.shape, N)
        for i in range(N):
            print("(%i/%i) Corrected (2) frame to H5" % (i+1,N))
            temp = ds_raw[i,:,:] - bg[:,:]
            row_medians = np.median(temp, axis=1)
            img2 = temp# - np.array([row_medians]).repeat(img.shape[1],axis=0).reshape((img.shape[0],img.shape[1]))
            ds_corr2[i,:,:] = img2[:,:]
            ds_corr2_sum[i] = img2.sum()
            ds_corr2_min[i] = img2.min()
            ds_corr2_max[i] = img2.max()
            ds_corr2_median[i] = np.median(img2)
    return filename_h5
            
def avi_to_h5(filename_avi, factor, keep_pngs=False):
    directory_frames = avi_to_frames(filename_avi)
    filename_h5 = frames_to_h5(directory_frames, factor=factor)
    filename_h5_new = os.path.abspath(os.path.dirname(filename_avi)) + "/" + (filename_avi.split("/")[-1])[:-len(".avi")] + ".cxi"
    os.system("mv %s %s" % (filename_h5,filename_h5_new))
    if not keep_pngs: os.system("rm -rf %s" % directory_frames)
    return filename_h5_new

def main():
    parser = argparse.ArgumentParser(description='Conversion of AVI to HDF5')
    parser.add_argument('filename', type=str, help='HDF5 filename', default=None)
    parser.add_argument('-k', '--keep_pngs', action="store_true", help='keep temporary PNG files')
    parser.add_argument('-f','--factor', type=float, help='multiply by factor', default=3700.)
    args = parser.parse_args()
    if args.filename is None:
        filenames = [f for f in os.listdir("./") if f.endswith(".avi")]
    else:
        filenames = [args.filename]
    for i,filename in enumerate(filenames):
        print(("(%i/%i)" % (i+1,len(filenames))), filename)
        avi_to_h5(filename, factor=args.factor, keep_pngs=args.keep_pngs)

if __name__ == "__main__":
    main()
