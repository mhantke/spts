import os
import numpy as np
import h5py
import scipy.interpolate

import itertools

import requests

from distutils.version import LooseVersion

import logging
logger = logging.getLogger('spts')

from matplotlib import pyplot as pypl
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

import seaborn as sns
sns.set_style("white")

import spts

from spts import config

adu_per_photon = {"Hamamatsu-C11440-22CU": 2.13}

pixel_size = {"Hamamatsu-C11440-22CU": 6.5E-6,
              "Fastcam": 20E-6}

# Transmissions at 
filter_transmission_530 = {
    "": 1.,
    "ND060":  21.18E-2,
    "ND090": 9.69E-2,
    "ND200": 0.97E-2,
    "ND300": 0.13E-2,
    "ND400": 0.01E-2,
}

evergreen_pulse_energy_L1 = {
    1: 8.20,
    2: 11.92,
    3: 16.28,
    4:	21.08,
    5:	25.80,
    6:	30.72,
    7:	36.52,
    8:	42.64,
    9:	48.52,
    10:	56.07,
    11:	61.82,
    12:	66.98,
    13:	72.29,
    14:	78.47,
    15:	84.07,
    16:	91.20,
    17:	98.55,
    18:	104.36,
    19:	109.60,
    20:	116.87,
}

evergreen_pulse_energy_L2 = {
    1: 7.20,
    2: 11.20,
    3: 15.04,
    4: 19.48,
    5: 24.88,
    6: 30.76,
    7: 35.72,
    8: 41.52,
    9: 47.68,
    10: 51.35,
    11:	59.64,
    12:	66.11,
    13:	72.00,
    14:	78.40,
    15:	84.51,
    16:	91.56,
    17:	98.33,
    18:	103.64,
    19:	109.75,
    20:	116.87,
}

PS_diameters_true = {
    20: 23.,
    40: 41.,
    60: 60.,
    70: 70.,
    80: 81.,
    100: 100.,
    125: 125.,
    220: 216.,
    495: 496., 
}

PS_diameters_uncertainties = {
    20: 2.,
    40: 4.,
    60: 4.,
    70: 3.,
    80: 3.,
    100: 3.,
    125: 3.,
    220: 4.,
    495: 8.,
}

PS_diameters_spread = {
    20: None,
    40: None,
    60: 10.2,
    70: 7.3,
    80: 9.5,
    100: 7.8,
    125: 4.5,
    220: 5.,
    495: 8.6,
}


# http://www.thinksrs.com/downloads/PDFs/ApplicationNotes/IG1pggasapp.pdf
_p_N_reading = [1.33322,
                2.66644,
                3.99966,
                5.33288,
                6.6661,
                7.99932,
                9.33254,
                10.66576,
                11.99898,
                13.3322,
                14.66542]

_p_He_actual = [1.33322*1.1,
                2.133152,
                2.66644,
                3.199728,
                3.599694,
                3.733016,
                3.99966,
                4.132982,
                4.266304,
                4.799592,
                6.6661]

p_He = lambda p_N_reading: (np.where(p_N_reading < 1.33322, 1.1 * p_N_reading, np.interp(p_N_reading, _p_N_reading, _p_He_actual)))
    
#pix_to_um = lambda x, navitar_zoom, objective_zoom_factor, pixel_size: x * (5./objective_zoom_factor) * 7.2E-3*np.exp(-navitar_zoom/3714.)/1024. * (pixel_size/20E-6)

# All in unit meter
def pix_to_m(camera_pixel_size, navitar_zoom, objective_zoom_factor):
    # Navitar Zoom Calibration. 5x Mitituyo objective insidet the vacuum chamber, coupled to outside navitar zoom lens connected to the fastcam.
    z_vs_p = np.array([[0, 7.0423E-6],
                       [1000, 5.3957E-6],
                       [2000, 4.1322E-6],
                       [3000, 3.112E-6],
                       [4000, 2.3659E-6],
                       [4600, 2.0202E-6],
                       [5000, 1.8007E-6],
                       [6000, 1.4045E-6],
                       [7000, 1.0593E-6],
                       [8000, 0.8104E-6],
                       [8923, 0.6313E-6]])
    p = np.interp(navitar_zoom, z_vs_p[:,0], z_vs_p[:,1])
    # Correct for different objective if necessary
    p *= 5./objective_zoom_factor
    # Correct for different camera pixel size if necessary
    p *= camera_pixel_size/20.E-6
    return p

def read_datasets_table_old(filename, D=None):
    if D is None:
        D = {}
    with open(filename, "r") as f:
        lines = f.readlines()
        if len(lines) > 1:
            pass
        #    lines = [l[:-1] for l in lines]
        else:
            lines = lines[0].split("\r")
    titles = lines[0].split(",")
    # Remove newline in last title
    titles[-1] = titles[-1][:-1]
    data_lines = lines[1:]
    for l in data_lines:
        L = l.split(",")
        # Remove newline in last item
        L[-1] = L[-1][:-1]
        name = L[titles.index("File")]
        D[name] = {}
        for t,d in zip(titles, L):
            D[name][t] = d
    return D

def read_datasets_table(filename=None, D=None, iddate=False):
    # Read from www or file
    if filename is None:
        document_id = "1mPW6QQLEtdYdEtvMktHsFL25FxNpozOXKRr6ibfrZnA"
        gid = "2081833535"
        url = "https://docs.google.com/spreadsheets/d/%s/export?format=tsv&gid=%s" % (document_id, gid)
        with requests.Session() as s:
            download = s.get(url)
            decoded_content = download.content.decode('utf-8')
        lines = decoded_content.split("\n")
    else:
        with open(filename, "r") as f:
            lines = f.readlines()

    # Initialise if necessary
    if D is None:
        D = {}

    # Clean up lines
    lines = [l for l in lines if l[0] != "#"]
    if len(lines) > 1:
        for i in range(len(lines)):
            while (lines[i].endswith("\n") or lines[i].endswith(" ") or lines[i].endswith("\r")):
                lines[i] = lines[i][:-1]
    else:
        lines = lines[0].split("\r")

    # Titles
    titles = lines[0].split("\t")
    for l in lines[1:3]:
        for i,s in enumerate(l.split("\t")):
            if len(s) > 0:
                titles[i] += " " + s

    # Interpret data type of all data fields and write into dict
    data_lines = lines[3:]
    for l in data_lines:
        L = l.split("\t")
        name = L[titles.index("File")]
        if iddate:
            name += "_" + L[titles.index("Date")]
        D[name] = {}
        for t,d in zip(titles, L):
            D[name][t] = estimate_type(d)
            
    return D

def find_datasets(D, option_name, option_value, decimals_precision=None):
    found_dataset_names = []
    for k,Dk in D.items():
        if option_name not in Dk:
            logger.error("Cannot find %s in D[%s]. Abort." % (option_name, k))
            return
        if (decimals_precision is None and Dk[option_name] == option_value) or \
           (decimals_precision is not None and (round(Dk[option_name], decimals_precision) == round(option_value, decimals_precision))):
            found_dataset_names.append(k)
    return found_dataset_names

def sort_datasets(D, option_name):
    values = []
    keys_list = D.keys()
    for k in keys_list:
        Dk = D[k]
        if option_name not in Dk:
            logger.error("Cannot find %s in D[%s]. Abort." % (option_name, k))
            return
        values.append(Dk[option_name])
    return [keys_list[i] for i in np.array(values).argsort()]
        
def estimate_type(var):
    #first test bools
    if var.lower() == 'true':
        return True
    elif var.lower() == 'false':
        return False
    elif var.lower() == 'none':
        return None
    else:
        #int
        try:
            return int(var)
        except ValueError:
            pass
        #float
        try:
            return float(var)
        except ValueError:
            pass
        #string
        try:
            return str(var)
        except ValueError:
            raise NameError('Something messed up autocasting var %s (%s)' % (var, type(var)))


def read_data(D, root_dir="/scratch/fhgfs/hantke/spts", only_scalars=False, skip_thumbnails=True, verbose=True, data_prefix="", iddate=False):
    ds_names = D.keys()
    ds_names.sort()
    for ds_name in ds_names:
        D[ds_name] = read_data_single(ds_name, D[ds_name], root_dir=root_dir, only_scalars=only_scalars,
                                      skip_thumbnails=skip_thumbnails, verbose=verbose, data_prefix=data_prefix, iddate=iddate)
    return D

def read_data_single(ds_name, D, root_dir="/scratch/fhgfs/hantke/spts", only_scalars=False, skip_thumbnails=True, verbose=True, data_prefix="", iddate=False):
    if iddate:
        fn_root = ds_name.split("_")[0]
    else:
        fn_root = ds_name
    if "Data Location" not in D.keys():
        folder = "%s%s/%s/%s_analysis" % (data_prefix, root_dir, D["Date"], fn_root)
    else:
        folder = "%s%s/%s_analysis" % (data_prefix, D["Data Location"], fn_root)

    D["filename"] = "%s/spts.cxi" % folder
    D["conf_filename"] = "%s/spts.conf" % folder 
        
    if not os.path.isfile(D["filename"]):
        if verbose:
            print("Skipping %s - file does not exist" % D["filename"])
    else:
        # Read data
        with h5py.File(D["filename"], "r") as f:
            version = '0.0.1' if '__version__' not in f else str(f['__version__'])
            if LooseVersion(version) > LooseVersion('0.0.1'):
                t_raw = "1_raw"
                t_process = "2_process"
                t_process_image = "image"
                t_denoise = "3_denoise"
                t_threshold = "4_threshold"
                t_detect =  "5_detect"
                t_analyse = "6_analyse"
                t_peak_prefix = "peak_"
                t_detect_dist_neighbor = "dist_neighbor"
                t_dislocation = "dislocation"
            else:
                t_raw = "1_measure" 
                t_process = "1_measure"
                t_process_image = "image"
                t_denoise = "1_measure"
                t_threshold = "1_measure"
                t_detect =  "2_detect"
                t_analyse = "3_analyse"
                t_peak_prefix = ""
                t_detect_dist_neighbor = "dists_neighbor"
                t_dislocation = None
            D["sat_n"] = np.asarray(f["/%s/saturated_n_pixels" % t_raw])
            D["x"]     = np.asarray(f["/%s/x" % t_detect])
            D["y"]     = np.asarray(f["/%s/y" % t_detect])
            sel = (D["x"] >= 0) * (D["y"] >= 0)
            D["area"]  = np.asarray(f["/%s/area" % t_detect])
            D["circumference"]  = np.asarray(f["/%s/%scircumference" % (t_analyse, t_peak_prefix)])
            D["eccentricity"]  = np.asarray(f["/%s/%seccentricity" % (t_analyse, t_peak_prefix)])                    
            D["dists_neighbor"] = np.asarray(f["/%s/%s" % (t_detect, t_detect_dist_neighbor)])
            D["sum"]   = np.asarray(f["/%s/%ssum" % (t_analyse, t_peak_prefix)])
            D["saturated"] = np.asarray(f["/%s/%ssaturated" % (t_analyse, t_peak_prefix)])
            if t_dislocation is not None:
                if ("/%s/%s" % (t_detect, t_dislocation)) in f:
                    D["dislocation"] = np.asarray(f["/%s/%s" % (t_detect, t_dislocation)])
            if "/5_detect/peak_score" in f:
                D["peak_score"] = np.asarray(f["/5_detect/peak_score"])
            D["merged"] = np.asarray(f["/%s/merged" % (t_detect)])
            t_peak_min = "%s/peak_min" % t_analyse
            t_peak_max = "%s/peak_max" % t_analyse
            t_peak_mean = "%s/peak_mean" % t_analyse
            t_peak_median = "%s/peak_median" % t_analyse
            for kp,tp in zip(["min", "max", "mean", "median"], [t_peak_min, t_peak_max, t_peak_mean, t_peak_median]):
                if tp in f:
                    D[kp] = np.asarray(f[tp])
            if not only_scalars:
                t_image = "/%s/%s" % (t_process, t_process_image)
                if t_image in f:
                    D["image1"] = np.asarray(f[t_image][1,:,:])
                    D["image2"] = np.asarray(f[t_image][2,:,:])
                if "/2_detect/image_labels" in f:
                    D["labels1"] = np.asarray(f["/%s/image_labels" % t_detect][1,:,:])
                t_thumbnails = ("/%s/peak_thumbnails" % (t_analyse))
                if t_thumbnails in f:
                    D["thumbnails1"] = np.asarray(f["%s/peak_thumbnails" % (t_analyse)][1,:,:,:])
                    if not skip_thumbnails:
                        D["thumbnails"] = []
                        for i in range(sel.shape[0]):
                            if sel[i].sum() == 0:
                                D["thumbnails"].append(np.asarray([]))
                            else:
                                tmp_shape = f["%s/peak_thumbnails" % (t_analyse)][i,0].shape
                                D["thumbnails"].append(np.asarray(f["%s/peak_thumbnails" % (t_analyse)][i, np.where(sel[i])[0], :, :]).reshape(sel[i].sum(), tmp_shape[0], tmp_shape[1]))
            if verbose:
                print("Read %s" % D["filename"])
        # Read config file        
        if os.path.isfile(D["conf_filename"]):
            D["conf"] = config.read_configfile(D["conf_filename"])
        else:
            if verbose:
                print("Skipping %s - file does not exist" % D["conf_filename"])
    return D

def mask_data(D, exclude_saturated_frames=False, verbose=True):
    keys = D.keys()
    keys.sort()
    for k in keys:
        x = D[k]["x"]
        y = D[k]["y"]
        # VALID PARTICLES
        v = (x > 0) * (y > 0)
        if exclude_saturated_frames:
            for i,sat_n in enumerate(D[k]["sat_n"]):
                if sat_n > 0:
                    v[i,:] = False
        D[k]["valid_particles"] = v
        # VALID PARTICLES / FRAME
        n_frame = np.array([v_frame.sum() for v_frame in v])
        D[k]["particles_per_frame"] = n_frame
        # NOT SATURATED
        n = D[k]["saturated"] == 0
        D[k]["not_saturated_particles"] = n
        n_fract = (n*v).sum()/float(v.sum())
        if "cx_focus" in D[k] and "cy_focus" in D[k] and "r_focus" in D[k]:
            # CENTERED PARTICLES (IN X AND Y, NOT IN Z)
            c = ((x-D[k]["cx_focus"])**2 + (y-D[k]["cy_focus"])**2) <= D[k]["r_focus"]**2
        else:
            c = np.ones(shape=v.shape, dtype='bool')
            if verbose:
                print("WARNING: No masking of centered particles for %s" % k)
        D[k]["centered_particles"] = c
        c_fract = (c*v).sum()/float(v.sum())
        # CIRCULAR PARTICLES
        circ = D[k]["circumference"]
        area = D[k]["area"]
        D[k]["circularity"] = (area/np.pi) / (np.finfo(np.float64).eps + (circ/(2*np.pi))**2)
        t = D[k]["circularity_threshold"] # 0.6 typically fine
        r = D[k]["circularity"] > t
        D[k]["circular_particles"] = r
        r_fract = (r*v).sum()/float(v.sum())
        # ISOLATED PARTICLES
        d_min = D[k]["dists_min"]
        i = D[k]["dists_neighbor"] >= d_min
        D[k]["isolated_particles"] = i
        i_fract = (i*v).sum()/float(v.sum())
        # SOME OUTPUT
        if verbose:
            print("%s: %.1f part. rate\t %i part.\t (%i %% not sat.; %i %% centered; %i %% circ.; %i %% isol.)" % (k, round(n_frame.mean(),1), v.sum(), 100.*n_fract, 100.*c_fract, 100.*r_fract, 100.*i_fract))
    return D

def read_thumbnails(D, k, index_mask, root_dir="/scratch/fhgfs/hantke/spts", Nmax=None):
    if "Data Location" not in D[k].keys(): 
        D[k]["filename"] = "%s/%s/%s_analysis/spts.cxi" % (root_dir,D[k]["Date"],k)           
    else:
        D[k]["filename"] = "%s/%s_analysis/spts.cxi" % (D[k]["Data Location"],k)           
    with h5py.File(D[k]["filename"], "r") as f:
        ni = index_mask.shape[0]
        nj = index_mask.shape[1]
        if ni != f["/3_analyse/thumbnails"].shape[0] or nj != f["/3_analyse/thumbnails"].shape[1]:
            print("ERROR: Shapes of index_mask (%s) and the h5dataset of thumbnails (%s) do not match!" % (str(index_mask.shape), str(f["/3_analyse/thumbnails"].shape)))
            return
        J,I = np.meshgrid(np.arange(nj), np.arange(ni))
        I = I[index_mask]
        J = J[index_mask]
        N = index_mask.sum()
        if Nmax is not None:
            N = min([Nmax, N]) 
        thumbnails = np.zeros(shape=(N, f["/3_analyse/thumbnails"].shape[2], f["/3_analyse/thumbnails"].shape[3]), dtype=f["/3_analyse/thumbnails"].dtype)
        for k,i,j in zip(range(N),I[:N],J[:N]):
            thumbnails[k,:,:] = np.asarray(f["/3_analyse/thumbnails"][i,j,:,:])
    return thumbnails

def fqls_to_mJ(fqls_us):
    tab_fqls = np.array([182, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
                         300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410], dtype=np.float64)
    tab_mJ   = np.array([53.60, 53.75, 53.25, 51.95, 49.98, 47.19, 44.30, 40.80, 37.30, 33.15, 29.38, 25.54,
                         21.58, 18.23, 15.18, 12.22,  9.65,  7.30,  5.37,  3.61,  2.33,  1.31,  0.58,  0.23], dtype=np.float64)
    if fqls_us < tab_fqls[0]:
        print("WARNING: given FQLS=%f out of range (min.: %f)" % (fqls_us, tab_fqls[0]))
        return tab_mJ[0]
    elif fqls_us > tab_fqls[-1]:
        print("WARNING: given FQLS=%f out of range (max.: %f)" % (fqls_us, tab_fqls[-1]))
        return tab_mJ[-1]
    else:
        f = scipy.interpolate.interp1d(tab_fqls, tab_mJ)
        return f(fqls_us)

def plot_focus(D, separate=False):
    keys = D.keys()
    keys.sort()
    if not separate:
        fig, (axs1, axs2, axs3) = pypl.subplots(3, len(keys), figsize=(2*len(keys),8))
    for i,k in enumerate(keys):
        if not separate:
            ax1 = axs1[i]
            ax2 = axs2[i]
            ax3 = axs3[i]
        else:
            fig, (ax1, ax2, ax3) = pypl.subplots(1, 3, figsize=(8,2))
        p_v = D[k]["valid_particles"]
        p_n = D[k]["not_saturated_particles"]
        p = p_v * p_n

        x = D[k]["x"]
        y = D[k]["y"]

        s = D[k]["sum"]
        
        cx = D[k]["cx_focus"]
        cy = D[k]["cy_focus"]
        r = D[k]["r_focus"]

        Nbins = 20
        xedges = np.linspace(0,1500,Nbins+1)
        yedges = np.linspace(0,1500,Nbins+1)

        N, xedges, yedges = np.histogram2d(y[p], x[p], bins=(xedges, yedges))
        I, xedges, yedges = np.histogram2d(y[p], x[p], bins=(xedges, yedges), weights=s[p])
        I = np.float64(I)/N

        # 2D histogram weighted by intensity
        ax1.imshow(I, interpolation="nearest", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap="hot", vmin=0)
        ax1.add_patch(Circle(
            (cx,cy),   # (x,y)
            r,          # width
            facecolor=(1,1,1,0),
            edgecolor=(0.5,0.5,0.5,1.),
            ls="--",
            lw=3.5,
        )
        )
        ax1.set_axis_off()
        ax1.set_title(k)

        # 2D histogram
        ax2.imshow(N, interpolation="nearest", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower')
        ax2.set_axis_off()
        
        #ax3.hist(y[p], bins=yedges, weights=s[p])
        s_median = np.zeros(Nbins)
        s_mean   = np.zeros(Nbins)
        s_min   = np.zeros(Nbins)
        s_max   = np.zeros(Nbins)
        for yi,y0,y1 in zip(range(Nbins), yedges[:-1], yedges[1:]):
            sel = (y0 <= y[p]) * (y[p] < y1)
            if sel.any():
                s_median[yi] = np.median(s[p][sel])
                s_mean[yi] = np.mean(s[p][sel])
                s_min[yi] = np.min(s[p][sel])
                s_max[yi] = np.max(s[p][sel])
            else:
                s_median[yi] = np.nan
                s_mean[yi] = np.nan
                s_min[yi] = np.nan
                s_max[yi] = np.nan

        #ax3.plot(yedges[:-1]+(yedges[1]-yedges[0])/2., s_min, color="black", ls='--')
        #ax3.plot(yedges[:-1]+(yedges[1]-yedges[0])/2., s_max, color="black", ls='--')                
        ax3.plot(yedges[:-1]+(yedges[1]-yedges[0])/2., s_median, color="blue")
        #ax3.plot(yedges[:-1]+(yedges[1]-yedges[0])/2., s_mean, color="green")
        ax3.axvline(cy-r, color=(0.5,0.5,0.5,1.), ls="--", lw=3.5)
        ax3.axvline(cy+r, color=(0.5,0.5,0.5,1.), ls="--", lw=3.5)

def plot_positions(D, separate=False, ylim=None, xlim=None, ilim=None, Nslices=5, Nbins=50, pngout=None, recenter=False, um=True):
    keys = D.keys()
    keys.sort()
    if not separate:
        fig, axs1 = pypl.subplots(2, len(keys), figsize=(2*len(keys), 6))
    for i,k in enumerate(keys):
        if not separate:
            (ax1, ax2, ax3, ax4, ax5, ax6) = axs1[i]
        else:
            fig, (ax1, ax2, ax3, ax4, ax5, ax6) = pypl.subplots(1, 6, figsize=(20,3))#, sharey=True)#, sharey=True)
            fig.suptitle(k)

        if "analysis" not in D[k]:
            D[k]["analysis"] = {}
        D[k]["analysis"]["plot_positions"] = {}
            
        intensity = D[k]["sum"]

        p_v = D[k]["valid_particles"]
        p_n = D[k]["not_saturated_particles"]
        p_p = intensity >= 0
        p_c = D[k]["circular_particles"]
        p = p_v * p_n * p_p * p_c

        if ilim is None:
            i0 = 0
            i1 = intensity.max()
        else:
            i0 = ilim[0]
            i1 = ilim[1]
            
        i = intensity >= i0
        i = intensity <= i1

        if um:
            camera_pixel_size = pixel_size[D[k]["Camera"]]
            navitar_zoom = D[k]["Zoom"]
            objective_zoom_factor = D[k]["Magn. Obj."]
            c = pix_to_m(camera_pixel_size=camera_pixel_size, navitar_zoom=navitar_zoom, objective_zoom_factor=objective_zoom_factor)/1E-6
        else:
            c = 1.

        x = D[k]["x"] * c
        y = D[k]["y"] * c

        if xlim is not None:
            xmin = xlim[0]
            xmax = xlim[1]
        else:
            xmin = 0 * c
            xmax = 1500 * c
            
        p *= x>=xmin
        p *= x<=xmax
            
        if ylim is not None:
            ymin= ylim[0]
            ymax= ylim[1]
        else:
            ymin = 0 * c
            ymax = 1500 * c

        p *= y>=ymin
        p *= y<=ymax
       
        xedges = np.linspace(xmin,xmax,Nbins+1)
        dx = (xedges[1]-xedges[0])

        yedges = np.linspace(ymin,ymax,Nslices+1)
        dy = (yedges[1]-yedges[0])
        ycenters = yedges[:-1] + dy/2.
        
        H = np.zeros(shape=(Nslices, Nbins))
        Hfit = np.zeros(shape=(Nslices, Nbins))
        
        results = {
            'success': np.zeros(Nslices, dtype='bool'),
            'A0': np.zeros(Nslices),
            'x0': np.zeros(Nslices),
            'sigma': np.zeros(Nslices),
            'fwhm': np.zeros(Nslices),
        }                    

        for si,y0,y1 in zip(range(Nslices), yedges[:-1], yedges[1:]):
            s = (y >= y0) * (y < y1) * i
            ydata, xdata = np.histogram(x[p*s], bins=xedges, normed=True)
            xdata = xdata[:-1] + (xdata[1]-xdata[0])/2.
            assert xdata.size == Nbins
            
            p_init = None
            p_result, nest = gaussian_fit(xdata=xdata,ydata=ydata,p_init=p_init)
            A0, x0, sigma = p_result

            if np.isfinite(A0) and np.isfinite(x0) and np.isfinite(sigma):
                results['success'][si] = True
                results['A0'][si] = A0
                results['x0'][si] = x0
                results['sigma'][si] = sigma
                results['fwhm'][si] = sigma * 2 * np.sqrt(2*np.log(2))
            
            H[si,:] = ydata
            Hfit[si,:] = nest

        xc = np.array(results['x0'][results['success']]).mean()
        if recenter:
            xic = abs(xdata-xc).argmin() 
            xi0 = xic-Nbins/8
            xi1 = xic+Nbins/8
            x0 = xdata[xi0]
            x1 = xdata[xi1]
            H = H[:,xi0:xi1+1]
            Hfit = Hfit[:,xi0:xi1+1]
        else:
            x0 = xmin + dx/2.
            x1 = xmax - dx/2.

        extent = [x0-dx/2., x1+dx/2., ymin, ymax]

        ax1.hist(intensity[p]**(1/6.), bins=100)
        ax1.axvline(i0**(1/6.))
        ax1.axvline(i1**(1/6.))

        top = 0
        tmp = p
        if tmp.sum() > 0:
            ax2.scatter(x[tmp], y[tmp], 2, color='blue')
            hx, foo = np.histogram(x[tmp], bins=xedges)
            p_result_out, nest = gaussian_fit(xdata=xdata,ydata=hx)#,p_init=p_init)
            ax3.plot(xdata, hx, color='blue')
            ax3.plot(xdata, nest, color='blue', ls='--')
            top = max([top, nest.max(), hx.max()])
        tmp = p*(i==1)
        if tmp.sum() > 0:
            ax2.scatter(x[tmp], y[tmp], 2, color='red')
            hx, foo = np.histogram(x[tmp], bins=xedges)
            p_result_in, nest = gaussian_fit(xdata=xdata,ydata=hx)#,p_init=p_init)
            ax3.plot(xdata, hx, color='red')
            ax3.plot(xdata, nest, color='red', ls='--')
            top = max([top, nest.max(), hx.max()])
        ax3.text(p_result_in[1], top*1.2, "FWHM=%.1f" % (p_result_in[2]*2*np.sqrt(2*np.log(2))), color='red', ha='center')
        ax3.text(p_result_out[1], top*1.4, "FWHM=%.1f" % (p_result_out[2]*2*np.sqrt(2*np.log(2))), color='blue', ha='center')
        ax3.set_ylim(0, top*1.6)

        D[k]["analysis"]["plot_positions"]["sigma"] = p_result_in[2]
        
        ax4.imshow(H, extent=extent, interpolation='nearest', origin='lower')        
        #ax4.plot(results['x0'][results['success']], ycenters[results['success']])
        #ax4.plot(results['x0'][results['success']]-results['fwhm'][results['success']]/2., ycenters[results['success']], color='red')
        #ax4.plot(results['x0'][results['success']]+results['fwhm'][results['success']]/2., ycenters[results['success']], color='red')
        ax4.set_title("Measurement")
        ax4.set_ylabel('y')

        ax5.imshow(Hfit, extent=extent, interpolation='nearest', origin='lower')
        ax5.plot(results['x0'][results['success']], ycenters[results['success']], color='green')
        ax5.plot(results['x0'][results['success']]-results['fwhm'][results['success']]/2., ycenters[results['success']], color='red')
        ax5.plot(results['x0'][results['success']]+results['fwhm'][results['success']]/2., ycenters[results['success']], color='red')
        ax5.set_title("Fit")
        
        ax6.plot(results['fwhm'][results['success']], ycenters[results['success']], color='red')
        fwhm_min = results['fwhm'][results['success']].min()
        fwhm_min_y = ycenters[results['success']][results['fwhm'][results['success']].argmin()]
        ax6.plot(fwhm_min, fwhm_min_y, 'o', color='red')
        ax6.text(0.8*fwhm_min, fwhm_min_y, "%.1f" % fwhm_min, ha='right')
        ax6.set_xlabel('FWHM')
        ax6.set_xlim(0, (x1-x0)/2.)
        ax6.set_aspect(1.0/2.)
        ax6.set_title("FWHM")
        
        for ax in [ax2, ax4, ax5]:
            ax.set_xlim(x0, x1)
            ax.set_xlabel('x')
        for ax in [ax2, ax4, ax5]:
            ax.set_aspect(1.0)
        for ax in [ax2, ax4, ax5, ax6]:
            ax.set_ylim(ymin, ymax)
            ax.xaxis.set_ticks([ax.xaxis.get_ticklocs()[0], ax.xaxis.get_ticklocs()[-1]])
            ax.yaxis.set_ticks([ymin, ymax])

        pypl.tight_layout()

        if pngout is None:
            pypl.show()
        else:
            if separate:
                tmp = "%s/%s.png" % (pngout, k)
            else:
                tmp = pngout
            fig.savefig(tmp)
            fig.clf()
            

def plot_circularities(D, separate=False):
    keys = D.keys()
    keys.sort()
    if not separate:
        fig, axs = pypl.subplots(1, len(keys), sharex='col', sharey='row', figsize=(3*len(keys),3))
    for i,k in enumerate(keys):
        if not separate:
            ax = axs[i]
        else:
            fig, ax = pypl.subplots(1, 1, figsize=(3,3))
        p_v = D[k]["valid_particles"]
        p_n = D[k]["not_saturated_particles"]
        p = p_v * p_n
        c = D[k]["circularity"]
        t = D[k]["circularity_threshold"]
        H = ax.hist(c[p], 100, normed=True, range=(0,2))
        ax.axvline(t, color="red", ls="--")
        ax.set_title(k)
        ax.set_xlabel("Circularity")

def plot_distances(D, separate=False):
    keys = D.keys()
    keys.sort()
    if not separate:
        fig, axs = pypl.subplots(1, len(keys), sharex='col', sharey='row', figsize=(3*len(keys),3))
    for i,k in enumerate(keys):
        if separate:
            fig, ax = pypl.subplots(1, 1, sharex='col', sharey='row', figsize=(3,3))
        else:
            ax = axs[i]
        p_v = D[k]["valid_particles"]
        p_n = D[k]["not_saturated_particles"]
        p = p_v * p_n
        d = D[k]["dists_neighbor"]
        d_min = D[k]["dists_min"]
        H = ax.hist(d[p], 100, normed=True, range=(0,2*d_min))
        ax.axvline(d_min, color="red", ls="--")
        ax.set_title(k)
        ax.set_xlabel("Distance [pixel]")

def plot_thumbnails(D, sizes=None, variable="size_nm", save_png=False):
    keys = D.keys()
    keys.sort()
    if sizes is None:
        sizes = np.arange(50, 625, 25)
    for k in keys:
        v = D[k]["valid_particles"] * D[k]["not_saturated_particles"] * D[k]["centered_particles"] * D[k]["circular_particles"] * D[k]["isolated_particles"]
        obj = D[k]["Objective"]
        fqls = float(D[k]["Laser FQLS"])
        Epulse = fqls_to_mJ(fqls)
        sum = D[k]["sum"] / Epulse
        size = sum**(1/6.) / 2.98 * 100.
        fig, axs = pypl.subplots(ncols=len(sizes)+2, figsize=((len(sizes)+2)*0.5,2.))
        fig.suptitle("%s - %s - %.1f mJ" % (obj, k, round(Epulse, 1)))
        shape = read_thumbnails(D, k, np.ones(v.shape, dtype=np.bool), Nmax=1)[0].shape
        for ax, si in zip(axs[2:], sizes):
            M = v*(abs(size - si) < 10)
            if M.sum():
                T = np.asarray(read_thumbnails(D, k, M, Nmax=1)[0], dtype=np.float64)
                if variable == "size_nm":
                    t = str(int(si.round()))
                    st = "%i%%" % int(round(100.*float(T.max())/65536.,0))
                    sst = "%.1e" % int(round(T.sum()/2.13, 1))
                    #sst = "%i" % int(round(T.sum()/2.13, 0))
                elif variable == "intensity_percent":
                    t = str(int(round(100.*T.max()/65500.)))
                    st = ""
                    sst = ""
                else:
                    print("ERROR: Argument variable=%s is invalid." % variable)
                    return
    
            else:
                T = np.zeros(shape=shape)
                if variable == "size_nm":
                    t = str(int(si.round()))
                    st = ""
                    sst = ""
                else:
                    t = ""
                    st = ""
                    sst = ""
            tn = "%i" % M.sum()
            ax.text(float(T.shape[1]-1)/2., -T.shape[0]-float(T.shape[0]-1)/2., tn, ha="center", va="center")
            ax.text(float(T.shape[1]-1)/2., -float(T.shape[0]-1)/2., t, ha="center", va="center")
            ax.text(float(T.shape[1]-1)/2., T.shape[0]+float(T.shape[0]-1)/2., st, ha="center", va="center")
            ax.text(float(T.shape[1]-1)/2., T.shape[0]*2.2+float(T.shape[0]-1)/2., sst, ha="center", va="center", rotation=45.)
            ax.imshow(T, interpolation="nearest", cmap="gnuplot", norm=LogNorm(0.1,65536))
            ax.set_axis_off()
        T = np.zeros(shape=shape)
        axs[0].imshow(T, interpolation="nearest", cmap="gnuplot", norm=LogNorm(0.1,65536))
        axs[0].text(0, -T.shape[0]-float(T.shape[0]-1)/2., "# particles", ha="left", va="center")
        axs[0].text(0, -float(T.shape[0]-1)/2., "Size [nm]", ha="left", va="center")
        axs[0].text(0, float(T.shape[0]-1)/2., "Image", ha="left", va="center")
        axs[0].text(0, T.shape[0]+float(T.shape[0]-1)/2., "Max./Sat.", ha="left", va="center")
        axs[0].text(0, T.shape[0]*2.2+float(T.shape[0]-1)/2., "Signal [ph]", ha="left", va="center")
        axs[0].set_axis_off()
        axs[1].set_axis_off()
        fig.subplots_adjust(hspace=5)
        if save_png:
            fig.savefig("./thumbnails_%s.png" % k, dpi=400)


def plot_hist(D, separate=False, projector=None, label="Scattering Intensity [adu]", Nbins=100, vmin=0, vmax=3E6, accumulate=False, fit=False, fit_p0=None, title=None, axvlines=None, particles_per_frame=False):
    
    keys = D.keys()
    keys.sort()
    
    if not separate and not accumulate:
        fig, axs = pypl.subplots(nrows=len(keys), ncols=1, figsize=(6,len(keys)*4), sharex=True)

    if accumulate:
        fig, axs = pypl.subplots(nrows=1, ncols=1, figsize=(6,4))
        data_acc = []

    ppf = np.array([])
        
    for i,k in enumerate(keys):
        
        if separate and not accumulate:
            fig, ax = pypl.subplots(nrows=1, ncols=1, figsize=(6,4))
        else:
            if len(keys) == 1 or accumulate:
                ax = axs
            else:
                ax = axs[i]
        
        p_v = D[k]["valid_particles"]
        p_n = D[k]["not_saturated_particles"]
        p_i = D[k]["isolated_particles"]
        p_c = D[k]["centered_particles"]
        m = p_v * p_n * p_i * p_c

        
        ppf = np.array(list(ppf) + list(D[k]["particles_per_frame"]))
        
        if projector is None:
            data = D[k]["sum"]
        else:
            data = projector(D[k]["sum"][m], D[k])        
            
        if accumulate:
            data_acc.extend(list(data))
            data = list(data_acc)
            
        if not accumulate or (i+1) == len(keys):
            H = np.histogram(data, bins=Nbins, range=(vmin, vmax))
            s = (H[1][:-1]+H[1][1:])/2.
            n = H[0]
            ax.fill_between(s, n, np.zeros_like(n), lw=0)#1.8)
            if separate or (i+1) == len(keys) or accumulate:
                ax.set_xlabel(label)
            ax.set_ylabel("Number of particles")
            sns.despine()

            if fit == 1:
                p_init = fit_p0
                p_result, nest = gaussian_fit(xdata=s,ydata=n,p_init=p_init)
                ax.plot(s, nest)
    
                ((A1, m1, s1)) = p_result
                ax.text(m1*1.1, A1*1.1, "%.3f (sigma = %.3f)" % (round(m1,3), round(s1,3)) , ha="left")
                ax.legend(["measured", "fit"])           
            
            if fit == 2:
                p_init = fit_p0
                p_result, nest = double_gaussian_fit(xdata=s,ydata=n,p_init=p_init)
                ax.plot(s, nest)
                ((A1, m1, s1), (A2, m2, s2)) = p_result
                ax.text(m1*1.1, A1*1.1, "%.3f (sigma = %.3f)" % (round(m1,3), round(s1,3)) , ha="left")
                ax.text(m2*1.1, A2*1.1, "%.3f (sigma = %.3f)" % (round(m2,3), round(s2,3)) , ha="left")
                ax.legend(["measured", "fit"])

            if axvlines is not None:
                ax.set_ylim((0, 1.2*n.max()))
                for pos,lab in axvlines:
                    ax.axvline(pos, ymin=0, ymax=0.9, color="black", ls="--")
                    ax.text(pos, 1.1*n.max(), lab, va="bottom", ha="center")

    if particles_per_frame:
        ax.text(1.0, 0.7, "%.1f particles / frame" % round(ppf.mean(),1), transform=ax.transAxes, ha="right")
                    
    if title is not None:
        fig.suptitle(title)

        
def plot_hist_intensity(D, separate=False, fqls_normed=False, Nbins=100, vmin=0, vmax=3E6, accumulate=False, fit=False, fit_p0=None, title=None, axvlines=None, particles_per_frame=False):
    if fqls_normed:
        projector = lambda s, Dk: s/adu_per_photon / fqls_to_mJ(float(Dk["Laser FQLS"]))
        label="Normed scattering Intensity [ph]"
    else:
        projector = lambda s, Dk: s/adu_per_photon
        label="Scattering Intensity [ph]"
    return plot_hist(D, separate=separate, projector=projector, label=label, Nbins=Nbins, vmin=vmin, vmax=vmax, accumulate=accumulate, fit=fit, fit_p0=fit_p0, title=title, axvlines=axvlines, particles_per_frame=particles_per_frame)   
    
def plot_hist_size(D, separate=False, fqls_normed=False, scaling_constant=33.56, Nbins=100, vmin=0, vmax=300, accumulate=False, fit=False, fit_p0=None, title=None, axvlines=None, particles_per_frame=False):
    projector = lambda s, Dk: (s / fqls_to_mJ(float(Dk["Laser FQLS"])))**(1/6.) * scaling_constant
    label="Particle diameter [nm]"
    return plot_hist(D, separate=separate, projector=projector, label=label, Nbins=Nbins, vmin=vmin, vmax=vmax, accumulate=accumulate, fit=fit, fit_p0=fit_p0, title=title, axvlines=axvlines, particles_per_frame=particles_per_frame)   
    
gaussian = lambda x,A0,x0,sigma: A0*np.exp((-(x-x0)**2)/(2.*sigma**2))
double_gaussian = lambda x,p1,p2: gaussian(x,p1[0],p1[1],p1[2])+gaussian(x,p2[0],p2[1],p2[2])

def double_gaussian_fit(xdata=None,ydata=None,p_init=None,show=False):
    from scipy.optimize import leastsq

    if xdata == None and ydata == None:
        # generate test data
        A01,A02 = 7.5,10.4
        mean1, mean2 = -5, 4.
        std1, std2 = 7.5, 4. 
        xdata =  np.linspace(-20, 20, 500)
        ydata = double_gaussian(xdata,[A01,mean1,std1],[A02,mean2,std2])

    if p_init == None:
        p_A01, p_A02, p_mean1, p_mean2, p_sd1, p_sd2  = [ydata.max(),
                                                         ydata.max(),
                                                         xdata[len(xdata)/3],
                                                         xdata[2*len(xdata)/3],
                                                         (xdata.max()-xdata.min())/10.,
                                                         (xdata.max()-xdata.min())/10.]
        p_init = [p_A01, p_mean1, p_sd1,p_A02, p_mean2, p_sd2] # Initial guesses for leastsq
    else:
        [p_A01, p_mean1, p_sd1,p_A02, p_mean2, p_sd2] = p_init # Initial guesses for leastsq

    err = lambda p,x,y: abs(y-double_gaussian(x,[p[0],p[1],p[2]],[p[3],p[4],p[5]]))

    plsq = leastsq(err, p_init, args = (xdata, ydata))

    p_result = [[A1, mean1, s1], [A2, mean2, s2]] = [[plsq[0][0],plsq[0][1],abs(plsq[0][2])],[plsq[0][3],plsq[0][4],abs(plsq[0][5])]]
    yest = double_gaussian(xdata,p_result[0],p_result[1])

    if show:
        import pylab
        pylab.figure()
        yinit = double_gaussian(xdata,[p_A01,p_mean1,p_sd1],[p_A02,p_mean2,p_sd2])
        yest1 = gaussian(xdata,plsq[0][0],plsq[0][1],plsq[0][2])
        yest2 = gaussian(xdata,plsq[0][3],plsq[0][4],plsq[0][5])
        pylab.plot(xdata, ydata, 'r.',color='red', label='Data')
        pylab.plot(xdata, yinit, 'r.',color='blue', label='Starting Guess')
        pylab.plot(xdata, yest, '-',lw=3.,color='black', label='Fitted curve (2 gaussians)')
        pylab.plot(xdata, yest1, '--',lw=1.,color='black', label='Fitted curve (2 gaussians)')
        pylab.plot(xdata, yest2, '--',lw=1.,color='black', label='Fitted curve (2 gaussians)')
        pylab.legend()
        pylab.show()
        
    return [p_result, yest]

def gaussian_fit(xdata=None,ydata=None,p_init=None,show=False):
    from scipy.optimize import leastsq
    
    if xdata is None and ydata is None:
        # Generate test data
        A0_test = 5.5
        x0_test = -2.
        s_test = 1. 
        xdata =  np.linspace(-20., 20., 500)
        ydata = gaussian(xdata,A0_test,x0_test,s_test)

    if p_init is None:
        # Guess parameters
        # A0
        A0_init = ydata.max()
        # x0
        x0_init = xdata[ydata.argmax()]
        # Sigma
        #s_init  = (xdata.max()-xdata.min())/10.
        dhalf = abs(ydata-ydata.max())
        left = xdata < x0_init
        right = xdata > x0_init
        fwhm_init = []
        if left.sum() > 0:
            fwhm_init.append(abs(xdata[left][dhalf[left].argmin()]-x0_init)*2)
        if right.sum() > 0:
            fwhm_init.append(abs(xdata[right][dhalf[right].argmin()]-x0_init)*2)       
        fwhm_init = np.asarray(fwhm_init).mean()
        s_init = fwhm_init / (2*np.sqrt(2*np.log(2)))
        # p_init
        p_init = [A0_init, x0_init, s_init]
    else:
        [A0_init, x0_init, s_init] = p_init

    g = lambda A0, x0, s: gaussian(xdata, A0, x0, s)
    err = lambda p: ydata-g(p[0],p[1],p[2])
    plsq = leastsq(err, p_init)
    
    p_result = [A0_result, x0_result, s_result] = [plsq[0][0],plsq[0][1],abs(plsq[0][2])]
    yest = gaussian(xdata,A0_result, x0_result, s_result)

    if show:
        import pylab
        pylab.figure()
        yinit = gaussian(xdata,A0_init, x0_init, s_init)
        pylab.plot(xdata, ydata, 'r.',color='red', label='Data')
        pylab.plot(xdata, yinit, 'r.',color='blue', label='Starting Guess')
        pylab.plot(xdata, yest, '-',lw=3.,color='black', label='Fitted curve')
        pylab.legend()
        pylab.show()
        
    return [p_result, yest]

def bootstrap_gaussian_fit(xdata,ydata,p_init0=None,show=False,Nfract=0.5,n=100, p_init_variation=0.5):
    if p_init0 == None:
        p_init = gaussian_fit(xdata,ydata)[0]
    else:
        p_init = p_init0
    ps = []
    N = int(round(len(xdata)*Nfract))
    for i in range(n):
        random_pick = np.random.choice(np.arange(xdata.size), size=N)
        xdata1 = xdata[random_pick]
        ydata1 = ydata[random_pick]
        p0 = np.array(p_init) * (1+((-0.5+np.random.rand(len(p_init)))*p_init_variation))
        ps.append(gaussian_fit(xdata1,ydata1,tuple(p0),show)[0])
    ps = np.array(ps)
    p_result = ps.mean(0)
    p_std = ps.std(0)
    yest = gaussian(xdata,p_result[0], p_result[1], p_result[2])
    return [p_result, yest, p_std]

def hist_gauss_fit(x, xmin, xmax, do_plot=False, ax=None, bootstrap=False, n_bootstrap=100, Nfract_bootstrap=0.75, p_init_variation_bootstrap=0., bins_max=500, bins_step=0.2):
    bins = 10
    success = False
    H, xedges = np.histogram(x, range=(xmin, xmax), bins=bins)           
    while (H>0.5*H.max()).sum() < 3:
        bins += 1
        H, xedges = np.histogram(x, range=(xmin, xmax), bins=bins)           
        if bins > bins_max:
            print("Fit failed.")
            break
    while not success:
        H, xedges = np.histogram(x, range=(xmin, xmax), bins=bins)           
        dxedges = xedges[1]-xedges[0]
        xcenters = xedges[:-1]+(dxedges)/2.
        if bootstrap:
            (A_fit, x0_fit, sigma_fit), H_fit, (A_fit_std, x0_fit_std, sigma_fit_std) = bootstrap_gaussian_fit(xcenters, np.asarray(H, dtype="float64"),
                                                                                                               n=n_bootstrap, Nfract=Nfract_bootstrap,
                                                                                                               p_init_variation=p_init_variation_bootstrap)
        else:
            (A_fit, x0_fit, sigma_fit), H_fit = gaussian_fit(xcenters, np.asarray(H, dtype="float64"))
        fwhm = abs(2*np.sqrt(2*np.log(2)) * sigma_fit)
        if fwhm/float(dxedges) > 5 and (not bootstrap or (np.array([A_fit_std, x0_fit_std, sigma_fit_std])/np.array([A_fit, x0_fit, sigma_fit]) < 1.).all()):
            success = True
        else:
            bins *= 1+bins_step
            bins = int(round(bins))

        if bins > bins_max:
            print("Fit failed.")
            break
    if do_plot:
        if ax is None:
            fig = pypl.figure(figsize=(2,3))
            _ax = fig.add_subplot(111)
        else:
            _ax = ax
        if success:
            _ax.plot(xcenters, H_fit)
        _ax.plot(xcenters, H, "o-")
        _ax.set_ylim(0, H.max()*1.1)

    if bootstrap:
        return (A_fit, x0_fit, sigma_fit), (xcenters, H, H_fit), (A_fit_std, x0_fit_std, sigma_fit_std), success
    else:
        return (A_fit, x0_fit, sigma_fit), (xcenters, H, H_fit), success


def calc_all_vecs(x, y, rmax=40):
    assert len(x) == len(y)
    n_frames = len(x)

    dx = []
    dy = []
    di1 = []
    di2 = []

    napp = 0
    
    s = (x > 0) * (y > 0)
    if not s.any():
        print("WARNING: Not a single point is valid.")
        return dx, dy, di1, di2

    i = np.arange(x.shape[1])
    
    for i_frame in range(n_frames):
        dx.append([])
        dy.append([])
        di1.append([])
        di2.append([])
        si = s[i_frame,:]
        ii = i[si]
        n = len(ii)
        xi = x[i_frame,:][si]
        yi = y[i_frame,:][si]
        combs = itertools.combinations(np.arange(n),2)
        for c1,c2 in combs:
            dxi = xi[c2]-xi[c1]
            dyi = yi[c2]-yi[c1] 
            ri = np.sqrt(dxi**2 + dyi**2)
            if ri <= rmax:
                dx[-1].append(dxi)
                dy[-1].append(dyi)
                di1[-1].append(ii[c1])
                di2[-1].append(ii[c2])
                napp += 1
        dx[-1] = np.asarray(dx[-1])
        dy[-1] = np.asarray(dy[-1])
        di1[-1] = np.asarray(di1[-1])
        di2[-1] = np.asarray(di2[-1])
    selfrac = float(napp) / float(s.sum())
    if selfrac < 0.3:
        print("WARNING: Selection fraction: %.2f%%" % (100*selfrac))
    return dx, dy, di1, di2

def calc_mean_vec(dx, dy, rmax=40, ds=1, dxmax=None, dx0_guess=None, dy0_guess=None):
    assert len(dx) == len(dy)
    dx_flat = list(itertools.chain.from_iterable(dx))
    dy_flat = list(itertools.chain.from_iterable(dy))
    assert len(dx_flat) == len(dy_flat)
    ranges = [[-rmax-0.5, rmax+0.5], [-0.5, rmax+0.5]]
    bins = [2*rmax/ds+1, rmax/ds+1]
    counts, xedges, yedges = np.histogram2d(dx_flat, dy_flat, bins=bins, range=ranges)
    counts = counts.swapaxes(1,0)
    X, Y = np.meshgrid(xedges[:-1] + (xedges[1]-xedges[0])/2.,
                       yedges[:-1] + (yedges[1]-yedges[0])/2.)
    if dxmax is not None:
        if dx0_guess is not None and dy0_guess is not None:
            counts[np.sqrt((X-dx0_guess)**2+(Y-dy0_guess)**2)>dxmax] = 0
        else:
            counts[abs(X)>dxmax] = 0
    i_max = counts.argmax()
    x_max = X.flat[i_max]
    y_max = Y.flat[i_max]
    return x_max, y_max

from scipy.ndimage.measurements import center_of_mass
def calc_com_vec(dx, dy, rmax=40, ds=1):
    assert len(dx) == len(dy)
    dx_flat = list(itertools.chain.from_iterable(dx))
    dy_flat = list(itertools.chain.from_iterable(dy))
    assert len(dx_flat) == len(dy_flat)
    ranges = [[-rmax-0.5, rmax+0.5], [-0.5, rmax+0.5]]
    bins = [2*rmax/ds+1, rmax/ds+1]
    counts, xedges, yedges = np.histogram2d(dx_flat, dy_flat, bins=bins, range=ranges)
    counts = counts.swapaxes(1,0)
    y_com, x_com = center_of_mass(counts)
    return x_com, y_com

def identify_pairs(dx, dy, di1, di2, dx0, dy0, dI=None, length_err_max=0.1, angle_deg_max=45., verbose=False):
    assert len(dx) == len(dy)
    n_frames = len(dx)
    di1_new = []
    di2_new = []
    dr0 = np.sqrt(dx0**2+dy0**2)
    nsuc = 0
    ntot = 0
    for i_frame, di1_i, di2_i, dx_i, dy_i in zip(range(n_frames), di1, di2, dx, dy):
        di1_new.append([])
        di2_new.append([])
        err_i = np.sqrt((dx_i-dx0)**2 + (dy_i-dy0)**2)/dr0
        aerr_i = np.arccos( (dx_i * dx0 + dy_i * dy0) / np.sqrt(dx_i**2 + dy_i**2) / np.sqrt(dx0**2 + dy0**2) ) / (2.*np.pi) * 360.
        sel = err_i <= length_err_max
        sel *= aerr_i <= angle_deg_max
        if dI is not None:
            sel *= dI[i_frame] < 0.25
        nsel = sel.sum()
        if nsel == 0:
            continue
        order = err_i.argsort()
        for i_pair in order[:nsel]:
            i1_pair = di1[i_frame][i_pair]
            i2_pair = di2[i_frame][i_pair]
            if (i1_pair not in di1_new[-1]) and (i1_pair not in di2_new[-1]) and (i2_pair not in di1_new[-1]) and (i2_pair not in di2_new[-1]):
                di1_new[-1].append(i1_pair)
                di2_new[-1].append(i2_pair)
            else:
                continue
        di1_new[-1] = np.asarray(di1_new[-1])
        di2_new[-1] = np.asarray(di2_new[-1])
        nsuc += len(di1_new[-1])
        ntot += nsel
    if verbose:
        print("success %i/%i" % (nsuc, ntot))
    return di1_new, di2_new

def filter_pairs(data, di1, di2, flat_output=False):
    data1 = []
    data2 = []
    for (i_frame, (di1_i, di2_i)) in enumerate(zip(di1, di2)):
        data1.append(data[i_frame, di1_i])
        data2.append(data[i_frame, di2_i])
    if flat_output:
        data1 = np.array(list(itertools.chain.from_iterable(data1)))
        data2 = np.array(list(itertools.chain.from_iterable(data2)))
    return  data1, data2

from scipy.optimize import least_squares
gaussian_beam = lambda y, y0, yR, w0: w0 * np.sqrt(1 + ((y-y0)/yR)**2)
def gaussian_beam_fit(y, fwhm, weights=1.):
    fwhm_min = fwhm.min() 
    fwhm_min_y = y[fwhm.argmin()]
    err = lambda v: (gaussian_beam(y, v[0], v[1], v[2]) - fwhm)*weights
    v0 = np.asarray([fwhm_min_y, 0.2, fwhm_min])
    if np.isnan(v0).any():
        return None, None, None
    else:
        y0_fit, yR_fit, w0_fit = least_squares(err, x0=v0).x
        return y0_fit, yR_fit, w0_fit
