#!/usr/bin/env python
import argparse
import os,sys
import OleFileIO_PL
from struct import unpack
import numpy as np

import sys
print sys.path

import h5writer
import spts
print "try"
import spts.camera
from spts.camera import CXDReader

def cxd_to_h5(filename_cxd, filename_bg_cxd, filename_cxi, Nbg_max):

    # Initialise reader(s)

    # Data
    print "Opening %s" % filename_cxd
    R = CXDReader(filename_cxd)

    # Background
    Rbg = None
    if filename_bg_cxd is not None:
        if filename_bg_cxd == filename_cxd:
            Rbg = R
        else:
            print "Opening %s" % filename_bg_cxd
            Rbg = CXDReader(filename_bg_cxd)
        # Collect background stack
        print "Collecting background frames..." 
        Nbg = min([Nbg_max, Rbg.get_number_of_frames()])
        for i in range(Nbg):
            frame = Rbg.get_frame(i)
            if i == 0:
                bg_stack = np.zeros(shape=(Nbg, frame.shape[0], frame.shape[1]), dtype=frame.dtype)
            bg_stack[i,:,:] = frame[:,:]
            print '(%d/%d) filling buffer for background estimation ...' % (i+1, Nbg)
        # Calculate median from background stack => background image    
        print "Calculating background estimate by median of buffer"
        bg = np.median(bg_stack, axis=0)
    else:
        bg = None

    # Initialise output CXI file
    W = h5writer.H5Writer(filename_cxi)

    # Initialise integration variables
    integrated_raw = None
    integrated_image = None
    integratedsq_raw = None
    integratedsq_image = None

    # Write frames
    N  = R.get_number_of_frames()
    for i in range(N):

        print '(%d/%d) Writing frame ...' % (i+1, N)

        frame = R.get_frame(i)
        image_raw = frame

        out = {}
        out["entry_1"] = {}

        # Raw data
        out["entry_1"]["data_1"] = {"data": image_raw}

        # Background-subtracted image
        if bg is not None:
            image_bgcor = np.asarray(frame, dtype=frame.dtype) - bg
            out["entry_1"]["image_1"] = {"data": image_bgcor}

        # Write to disc
        W.write_slice(out)
            
        if integrated_raw is None:
            integrated_raw = np.zeros(shape=frame.shape, dtype='float64')
        if integratedsq_raw is None:
            integratedsq_raw = np.zeros(shape=frame.shape, dtype='float64')
        integrated_raw += np.asarray(image_raw, dtype='float64')
        integratedsq_raw += np.asarray(image_raw, dtype='float64')**2

        if bg is not None:
            if integrated_image is None:
                integrated_image = np.zeros(shape=frame.shape, dtype='float64')
            if integratedsq_image is None:
                integratedsq_image = np.zeros(shape=frame.shape, dtype='float64')
            integrated_image += image_bgcor
            integratedsq_image += np.asarray(image_bgcor, dtype='f')**2
        
    # Write integrated images
    print 'Writing integrated images...'
    out = {"entry_1": {"data_1": {}, "image_1": {}}}
    if integrated_raw is not None:
        out["entry_1"]["data_1"]["data_mean"] = integrated_raw / float(N)
    if integrated_image is not None:
        out["entry_1"]["image_1"]["data_mean"] = integrated_image / float(N)
    if integratedsq_raw is not None:
        out["entry_1"]["data_1"]["datasq_mean"] = integratedsq_raw / float(N)
    if integratedsq_image is not None:
        out["entry_1"]["image_1"]["datasq_mean"] = integratedsq_image / float(N)
    W.write_solo(out)

    print "Closing files..."
    # Close CXI file    
    W.close()
    # Close readers
    R.close()
    if Rbg is not None and not Rbg.is_closed():
        Rbg.close()
    print "Clean exit."

if __name__ == "__main__":

    ### If there is an uppercase "-" in an optional argument it is converted internally to a "_", so we acccess these arguments in code with a "_" as well!!! 
    parser = argparse.ArgumentParser(description='Conversion of CXD (Hamamatsu file format) to HDF5') # positional argument
    parser.add_argument('filename', type=str, help='CXD filename') # optional argument
    parser.add_argument('-b','--background-filename', type=str, help='CXD filename with photon background data') # optional argument
    parser.add_argument('-o','--out-filename', type=str, help='destination file') # optional argument
    parser.add_argument('-n','--bg-frames-max', type=int, help='maximum number of frames that will be used for background calculation', default=250) # optional argument
    
    args = parser.parse_args()

    f = args.filename

    ### If background file is supplied, do background subtraction - if not supplied, proceed as usual!!!
    BackgroundFileExists = bool(args.background_filename)
    if BackgroundFileExists == False: 
        f_bg = None 
    elif BackgroundFileExists == True:
        f_bg = args.background_filename
    else: 
        f_bg = f[:-4] + "_bg.cxd"
        if not os.path.isfile(f_bg):
            print "WARNING: File with background frames not found in location %s.\n\tFalling back to determining background from the real data frames." % f_bg
            f_bg = f

    ### If the output file name is not specified the original file name is used and .cxi is appended to it!!!
    if args.out_filename:
        f_out = args.out_filename
    else:
        f_out = f[:-4] + ".cxi"
            
    ### Checks whether given input file is in the correct .cxd format!!!
    ### Call cxd_to_h5(); with f = filename_cxd, f_bg = filename_bg_cxd, f_out = filename_cxi, and arg.bg_frames_max = Nbg_max!!!
    if not args.filename.endswith(".cxd"):
        print "ERROR: Given filename %s does not end with \".cxd\. Wrong format!" % args.filename
    else:
        cxd_to_h5(f, f_bg, f_out, args.bg_frames_max)
