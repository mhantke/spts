#!/usr/local/bin/python3
import argparse
import os, sys, shutil
import time

import socket

import logging

import spts.log
from spts.log import log_and_raise_error,log_warning,log_info,log_debug
logger = spts.logger

import spts.config
import spts.worker

# Add SPTS stream handler to other loggers
import h5writer 
#h5writer.logger.addHandler(spts._stream_handler)
import mulpro
#mulpro.logger.addHandler(spts._stream_handler)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mie scattering imaging data analysis')
    parser.add_argument('-v', '--verbose', dest='verbose',  action='store_true', help='verbose mode', default=True)
    parser.add_argument('-d', '--debug', dest='debug',  action='store_true', help='debugging mode (even more output than in verbose mode)', default=False)
    parser.add_argument('-c','--cores', type=int, help='number of cores', default=1)
    parser.add_argument('-m','--mpi', dest='mpi', action='store_true', help='mpi processes = reader(s) + writer', default=False)
    args = parser.parse_args()

    if not os.path.exists("./spts.conf"):
        parser.error("Cannot find configuration file \"spts.conf\" in current directory.")
    lvl = logging.WARNING
    if args.verbose:
        lvl = logging.INFO
    if args.debug:
        lvl = logging.DEBUG
    spts.logger.setLevel(lvl)
    h5writer.logger.setLevel(lvl)
    logging.basicConfig(level=lvl)
    
    log_info(logger, "Hostname: %s" % socket.gethostname())
        
    conf = spts.config.read_configfile("./spts.conf")

    if args.mpi and args.cores > 1:
        parser.error("Specifying cores > 1 is only permitted when not running with MPI. ")
    
    if args.mpi:
        import mpi4py
        comm = mpi4py.MPI.COMM_WORLD
        is_worker = comm.rank > 0
        H = h5writer.H5WriterMPISW("./spts.cxi", comm=comm, chunksize=100, compression=None)
        if is_worker:
            W = spts.worker.Worker(conf, i0_offset=comm.rank-1, step_size=comm.size-1)
    else:
        is_worker = True
        H = h5writer.H5Writer("./spts.cxi")
        W = spts.worker.Worker(conf)

    if is_worker:
        if args.cores > 1:
            mulpro.mulpro(Nprocesses=args.cores-1, worker=W.work, getwork=W.get_work, logres=H.write_slice)
        else:
            while True:
                t0 = time.time()
                log_debug(logger, "Read work package (analysis)")
                w = W.get_work()
                if w is None:
                    log_debug(logger, "No more images to process")
                    break
                log_debug(logger, "Start work")
                l = W.work(w)
                t1 = time.time()
                t_work = t1-t0
                t0 = time.time()
                H.write_slice(l)
                t1 = time.time()
                t_write = t1-t0
                log_info(logger, "work %.2f sec / write %.2f sec" % (t_work, t_write))            

    H.write_solo({'__version__': spts.__version__})
    H.close()

    log_info(logger, "SPTS - Clean exit.")
                
            
