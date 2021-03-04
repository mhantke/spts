import itertools
import matplotlib
#matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as ppl
import h5py
import os

import config

pix_to_m = lambda x, navitar_zoom, objective_zoom_factor, pixel_size: x * (5./objective_zoom_factor) * 7.2E-3*np.exp(-navitar_zoom/3714.)/1024. * (pixel_size/20E-6)

def get_run(timestamp=None):
    D = {}
    if timestamp is None:
        D["root"] = "."
        D["timestamp"] = os.path.abspath(D["root"]).split("/")[-1][:-len("_analysis")]
    else:
        D["root"] = "/scratch/fhgfs/hantke/spts/%s/%s_analysis" % (timestamp.split("_")[0],timestamp)
        D["timestamp"] = timestamp
    D["data_filename"] = "%s/spts.cxi" % D["root"]
    D["conf_filename"] = "/home/hantke/src/spts/detect_particles/spts.conf"
    #D["conf_filename"] = "%s/spts.conf" % D["root"]
    if not os.path.exists(D["conf_filename"]):
        print("ERROR: Cannot find configuration file \"%s\"." % D["conf_filename"])
        return
    if not os.path.exists(D["data_filename"]):
        print("ERROR: Cannot find data file \"%s\"." % D["data_filename"])
        return
    D["conf"] = config.read_configfile(D["conf_filename"])    
    return D
    
def get_data(timestamp=None):
    D = get_run(timestamp)
    conf = D["conf"]
    
    with h5py.File(D["data_filename"], "r") as f:
        D["x"]               = np.asarray(f["/x"])
        D["y"]               = np.asarray(f["/y"])
        D["peak_sum"]        = np.asarray(f["/peak_sum"])
        D["peak_bg_corners"] = np.asarray(f["/peak_bg_corners"])
        D["sat"]             = np.asarray(f["/saturated"])
        D["n"]               = np.asarray(f["/n"])
        N_frames             = len(D["x"])

    # Exclude blanks
    valid = (D["x"]>0) * (D["y"]>0)
    #valid = valid * (D["peak_sum"]>=0)

    # Exclude saturated
    valid = valid * (D["sat"]==0)    
    if conf["eval"]["exclude_saturated_frames"]:
        # Exclude all particles in saturated frame
        for i_frame in range(N_frames):
            if (D["sat"][i_frame,:] > 0).any():
                valid[i_frame,:] = False
                
    D["cx"] = np.median(D["x"][valid])
    D["cy"] = np.median(D["y"][valid])

    # Rescale
    D["x_m"] = pix_to_m(D["x"], conf["eval"]["zoom"], conf["eval"]["objective"], conf["eval"].get("pixel_size", 20E-6))
    D["y_m"] = pix_to_m(D["y"], conf["eval"]["zoom"], conf["eval"]["objective"], conf["eval"].get("pixel_size", 20E-6))
    D["cx_m"] = pix_to_m(D["cx"], conf["eval"]["zoom"], conf["eval"]["objective"], conf["eval"].get("pixel_size", 20E-6))
    D["cy_m"] = pix_to_m(D["cy"], conf["eval"]["zoom"], conf["eval"]["objective"], conf["eval"].get("pixel_size", 20E-6))

    # Restrict to focus area
    if conf["eval"]["focus_size_m"] > 0:
        valid *= abs(D["x_m"] - D["cx_m"]) <= conf["eval"]["focus_size_m"]/2.
        valid *= abs(D["y_m"] - D["cy_m"]) <= conf["eval"]["focus_size_m"]/2.
        
    # Valid / saturated particles per frame
    D["n_valid"] = np.array([v.sum() for v in valid])
    D["n_sat"]   = np.array([v.sum() for v in D["sat"]])
    
    # Add validity array
    D["valid"] = valid

    return D




def plot_beam_width(pos, bins_edges, filename, zoom_navitar, title=""):
    fig = ppl.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    bins_center = bins_edges[:-1] + (bins_edges[1] - bins_edges[0])/2.
    n = ax.hist(pos,bins=bins_edges)[0]
    gauss = lambda x, A, x0, sigma: A*np.exp(-(x-x0)**2/(2.*sigma**2))
    err = lambda v: gauss(bins_center, v[0], v[1], v[2]) - n
    A,x0,sigma = scipy.optimize.leastsq(err, [n.max(), bins_center[n.argmax()], pos.std()/5.])[0]
    fit_x = np.linspace(bins_center[0], bins_center[-1], 500)
    fit_y = gauss(fit_x, A, x0, sigma)
    fwhm  = 2*np.sqrt(2*np.log(2))*sigma
    ax.plot(fit_x, fit_y)
    #ax.plot(bins_center, n)
    ax.text(x0+3*sigma,2*A/3.,"sigma = %.1f um" % np.round(sigma,1), ha="left")
    ax.text(x0+3*sigma,A/3.,"FWHM = %.1f um" % np.round(fwhm,1), ha="left")
    fig.savefig(filename, dpi=dpi)
    ppl.clf()
    return fwhm

def circle(x,y,d,Nx,Ny):
    X,Y = np.meshgrid(np.arange(Nx),np.arange(Ny))
    X = X-x
    Y = Y-y
    R = np.sqrt(X**2+Y**2)
    C = abs(R-d/2.) < 1.
    return C

def output1(image, params, output_dir, i):
    fig = ppl.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap="binary",vmin=0,vmax=threshold*2)
    for xi,yi,sumval_i in zip(params["x"],params["y"],params["sum"]):
        c = circle(xi,yi,d_integrate,img.shape[1],img.shape[0])
        ax.imshow(np.log10(c*10),cmap="winter")
        c = circle(xi,yi,2*d_integrate,img.shape[1],img.shape[0])
        ax.imshow(np.log10(c*10),cmap="winter")
        ax.text(xi+d_integrate/2.+5,yi,"x=%.1f y=%.1f sum=%i" % (xi.round(1),yi.round(1),sumval_i.round(0)), va="center", ha="left")
        if yi > (4*d_integrate):
            yi_reference = yi-2*d_integrate
        else:
            yi_reference = yi+2*d_integrate
        #if full_output:
        #    c = circle(xi,yi_reference,d_integrate,img.shape[1],img.shape[0])
        #    ax.imshow(np.log10(c*10),cmap="summer")
        #    c = circle(xi,yi_reference,2*d_integrate,img.shape[1],img.shape[0])
        #    ax.imshow(np.log10(c*10),cmap="summer")
        #    ax.text(xi+d_integrate/2.+5,yi_reference,"sum=%i" % (sumval_i.round(0)), va="center", ha="left")
    ppl.savefig("%s/particles%09i.png" % (output_dir,i))
    ppl.clf()

def output2(image, image_smooth, image_binary, params, output_dir, i):
    ppl.imsave("%s/image_binary_%09i.png" % (output_dir,i), image_binary)
    ppl.imsave("%s/image_smooth_%09i.png" % (output_dir,i), image_smooth, vmin=0, vmax=threshold)
    
