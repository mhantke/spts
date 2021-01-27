#!/usr/bin/env python
import numpy as np
import scipy.optimize
import scipy.ndimage

import logging
logger = logging.getLogger(__name__)

import log
from log import log_debug, log_info, log_warning, log_and_raise_error

THUMBNAILS_WINDOW_SIZE_DEFAULT = 30

def analyse_particles(image, image_raw, saturation_mask, i_labels, labels, x, y, merged, n_particles_max, full_output, **conf_analysis):

    ########## Test manual treshold
    #image[image < 0] = 0

    image_raw[image_raw < 0] = 0 
    ########## Test manual treshold

    if full_output:            
        masked_image = np.zeros_like(image_raw)
        if conf_analysis["integration_mode"] == "windows":
            thumbnails = np.zeros(shape=(n_particles_max, conf_analysis["window_size"], conf_analysis["window_size"]), dtype=image.dtype) 
        else:
            thumbnails = np.zeros(shape=(n_particles_max, THUMBNAILS_WINDOW_SIZE_DEFAULT, THUMBNAILS_WINDOW_SIZE_DEFAULT), dtype=image.dtype) 
    else:
        masked_image = None 
        thumbnails = None 
    mask = None 

    N = len(i_labels) 

    psuccess = np.zeros(N, dtype='bool') 
    psum  = np.zeros(N) - 1 
    pmin  = np.zeros(N) - 1 
    pmax  = np.zeros(N) - 1 
    pmean  = np.zeros(N) - 1 
    pmedian  = np.zeros(N) - 1 

    psat  = np.zeros(N) - 1 
    
    psize = np.zeros(N) - 1 
    pecc  = np.zeros(N) - 1 
    pcir  = np.zeros(N) - 1 
    
    for i, i_label, x_i, y_i, m in zip(range(N), i_labels, x, y, merged): 

        if x_i < 0 or y_i < 0: 
            continue 
        
        # Analyse
        pixels = i_label == labels 
        psat[i]  = (pixels * saturation_mask).any() 
        psize[i] = pixels.sum() 
        pecc[i] = measure_eccentricity(intensity=image*pixels, mask=pixels) 
        pcir[i] = measure_circumference(pixels) 
        
        if conf_analysis["integration_mode"] == "windows": 
            values = get_values_window(image_raw, int(round(x_i)), int(round(y_i)), window_size=conf_analysis["window_size"], circle_window=conf_analysis["circle_window"], i=i, 
                                       masked_image=masked_image, thumbnails=thumbnails) 
        elif conf_analysis["integration_mode"] == "labels": 
            values = get_values_label(image_raw, labels, i_label, 
                                      masked_image=masked_image, thumbnails=thumbnails) 
        else:
            log_and_raise_error(logger, "%s is not a valid integration_mode!" % conf_analysis["integration_mode"]) 

        if values is not None and len(values) > 0:
            psuccess[i] = True
            psum[i] = values.sum()
            pmin[i] = values.min()
            pmax[i] = values.max()
            pmean[i] = values.mean()
            pmedian[i] = np.median(values)
            
    success = True
    return success, psuccess, psum, pmean, pmedian, pmin, pmax, psize, psat, pecc, pcir, masked_image, thumbnails

def get_values_window(image, x_i, y_i, window_size, circle_window, i, masked_image=None, thumbnails=None):

    if not window_size % 2:
        log_and_raise_error(logger, "window_size (%i) must be an odd number. Please change your configuration and try again." % window_size)
        return None
    
    if (x_i-window_size/2 < 0) or \
       x_i-window_size/2+window_size>=image.shape[1] or \
       (y_i-window_size/2 < 0) or \
       y_i-window_size/2+window_size>=image.shape[0]:
        log_info(logger, "Particle too close to edge - cannot be analysed.")
        return None

    values = image[y_i-(window_size-1)/2:y_i+(window_size-1)/2+1,
                   x_i-(window_size-1)/2:x_i+(window_size-1)/2+1]
    
    if circle_window:
        mask = make_circle_mask(window_size)
        values = values[mask]
        
        if masked_image is not None:
            (masked_image[y_i-(window_size-1)/2:y_i+(window_size-1)/2+1,
                          x_i-(window_size-1)/2:x_i+(window_size-1)/2+1])[mask] = values[:]
        if thumbnails is not None:
            (thumbnails[i,:,:])[mask] = values[:]
    else:
        if masked_image is not None:
            masked_image[y_i-(window_size-1)/2:y_i+(window_size-1)/2+1,
                         x_i-(window_size-1)/2:x_i+(window_size-1)/2+1] = values[:,:]
        if thumbnail is not None:
            thumbnails[i,:,:] = values[:,:]

    return values
    
def get_values_label(image, labels, i_label, masked_image=None, thumbnails=None):

    pixels = i_label==labels
    values = image[pixels]
    if masked_image is not None:
        masked_image[pixels] = values[:]
    if thumbnails is not None:
        i = (image * pixels).argmax()
        x = i % image.shape[1]
        y = i / image.shape[1]
        n = thumbnails.shape[1]
        Ny = image.shape[0]
        Nx = image.shape[1]
        xmin = x-n/2
        xmin = xmin if xmin > 0 else 0
        ymin = y-n/2
        ymin = ymin if ymin > 0 else 0
        xmax = x-n/2+n
        xmax = xmax if xmax < Nx else Nx
        ymax = y-n/2+n
        ymax = ymax if ymax < Ny else Ny
        tmp = image[ymin:ymax, xmin:xmax]
        thumbnails[i_label,:tmp.shape[0],:tmp.shape[1]] = tmp[:,:] 
    return values
    

def measure_eccentricity(intensity, mask):
    if intensity.sum() > 0 and mask.sum() > 0:
        com_intensity = scipy.ndimage.measurements.center_of_mass(intensity)
        com_mask      = scipy.ndimage.measurements.center_of_mass(mask)
        off = np.sqrt((com_intensity[0]-com_mask[0])**2+(com_intensity[1]-com_mask[1])**2)
        return off
    else:
        return -1

def measure_circumference(mask):
    hor = scipy.ndimage.sobel(mask, 0)
    ver = scipy.ndimage.sobel(mask, 1)
    gra = np.hypot(hor, ver)
    border = (gra != 0) * mask
    return border.sum()

def make_circle_mask(diameter):
    center = (diameter-1.)/2.
    x,y = np.meshgrid(np.arange(diameter), np.arange(diameter))
    rsq = (x-center)**2 + (y-center)**2
    mask = rsq <= center**2
    return mask
