#!/usr/local/bin/python3
import numpy as np
import itertools
import scipy.ndimage.measurements

import logging
logger = logging.getLogger(__name__)

import log
from log import log_debug, log_info, log_warning, log_and_raise_error

def find_particles(image_scored, image_thresholded, min_dist, n_particles_max, peak_centering="center_of_mass"):

    n_lit = image_thresholded.sum()
    log_debug(logger, "%i pixels above threshold" % n_lit)

    success = False
    return_default = success, [], None, None, None, None, None, None, None, None
        
    if  n_lit == 0:
        return return_default

    else:
        log_debug(logger, "Label image")
        labels, n_labels = scipy.ndimage.measurements.label(image_thresholded)
        i_labels = range(1, n_labels+1)
        
        if n_labels > n_particles_max:
            log_info(logger, "%i labels - (frame overexposed? too many particles?), skipping analysis" % n_labels)
            return return_default

        V = [image_scored[i_label == labels].max() for i_label in i_labels]

        nx = image_thresholded.shape[1]
        ny = image_thresholded.shape[0]
        x,y = np.meshgrid(np.arange(nx), np.arange(ny))
        x = x[image_thresholded]    
        y = y[image_thresholded]
        l = labels[image_thresholded]
        v = image_scored[image_thresholded]
        
        if peak_centering == "center_of_mass":
            log_debug(logger, "Determine centers of mass")
            com = scipy.ndimage.measurements.center_of_mass(image_thresholded, labels, i_labels)
            X = [com_x for com_y,com_x in com]
            Y = [com_y for com_y,com_x in com]
            
        elif peak_centering == "center_to_max":
            log_debug(logger, "Determine maximum positions")
            X = []
            Y = []
            for i_label in i_labels:
                i_max = (v*(l == i_label)).argmax()
                X.append(x[i_max])
                Y.append(y[i_max])
        else:
            log_and_raise_error(logger, "%s is not a valid argument for peak_centering!" % peak_centering)    
            
        dislocation = []
        for i_label, xc, yc in zip(i_labels, X, Y):
            i_max = (v*(l == i_label)).argmax()
            dislocation.append(np.sqrt((x[i_max]-xc)**2 + (y[i_max]-yc)**2))

        merged = list(np.zeros(len(i_labels), dtype=np.bool))
        
        # Merge too close peaks
        log_debug(logger, "Merge too close points")
        i_labels, labels, X, Y, V, merged, dists_closest_neighbor = merge_close_points(i_labels, labels, X, Y, V, merged, min_dist)
        log_debug(logger, "Clean up labels")
        i_labels, labels = clean_up_labels(i_labels, labels)
        n_labels = len(i_labels)            
        
        log_debug(logger, "Measure size of each peak")
        areas = measure_areas(i_labels, labels)
        
        success = True
        areas  = np.asarray(areas)
        X      = np.asarray(X)
        Y      = np.asarray(Y)
        dislocation = np.asarray(dislocation)
        V      = np.asarray(V)
        merged = np.asarray(merged)
        dists_closest_neighbor = np.asarray(dists_closest_neighbor)
        
        return success, i_labels, labels, areas, X, Y, V, merged, dists_closest_neighbor, dislocation


def measure_areas(i_labels, labels):
    areas = []
    for label in i_labels:
        a = (labels == label).sum()
        assert a > 0
        areas.append(a)
    return areas

def merge_close_points(i_labels, labels, X, Y, V, merged, min_dist):
    
    while True:

        n_labels = len(i_labels)
        combs = np.asarray([c for c in itertools.combinations(range(n_labels),2)])
        dists = np.asarray([np.sqrt((X[c[0]]-X[c[1]])**2+(Y[c[0]]-Y[c[1]])**2) for c in combs])

        too_close = dists<min_dist
        
        if too_close.sum() > 0:
            
            dists = dists[too_close]
            combs = combs[too_close]
            combs = combs[dists.argsort()]
            closest_comb = combs[0]

            # Merge positions by averaging
            X[closest_comb[0]] = (X[closest_comb[0]] + X[closest_comb[1]]) / 2
            Y[closest_comb[0]] = (Y[closest_comb[0]] + Y[closest_comb[1]]) / 2.
            V[closest_comb[0]] = max([V[closest_comb[0]], Y[closest_comb[1]]])
            X.pop(closest_comb[1])
            Y.pop(closest_comb[1])
            V.pop(closest_comb[1])
            # Replace label
            labels[labels==i_labels[closest_comb[1]]] = i_labels[closest_comb[0]]
            i_labels.pop(closest_comb[1])
            # Update merged flag list
            merged[closest_comb[0]] = True
            merged.pop(closest_comb[1])

            N = len(i_labels)
            assert len(X) == N
            assert len(Y) == N
            assert len(merged) == N
            assert len(V) == N

        else:

            dists_closest_neighbor = np.zeros(n_labels)
            if n_labels == 1:
                dists_closest_neighbor[0] = -1
            else:
                for i0 in range(n_labels):
                    dists_closest_neighbor[i0] = np.asarray([np.sqrt((X[i0]-X[i1])**2+(Y[i0]-Y[i1])**2) for i1 in range(n_labels) if i1 != i0]).min()
                    
            return i_labels, labels, X, Y, V, merged, dists_closest_neighbor

            
def clean_up_labels(i_labels, labels):
    labels_new = np.zeros_like(labels)
    i_labels_new = range(1, len(i_labels)+1)
    for i_label_new,i_label_old in zip(i_labels_new, i_labels):
        labels_new[labels == i_label_old] = i_label_new
    return i_labels_new, labels_new
    
        
def test_distances(X, Y, min_dist):
    N = len(X)
    combs = np.asarray([c for c in itertools.combinations(range(N),2)])
    dists = np.asarray([np.sqrt((X[c[0]]-X[c[1]])**2+(Y[c[0]]-Y[c[1]])**2) for c in combs])
    too_close = dists<min_dist
    assert too_close.sum() == 0
