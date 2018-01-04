import numpy as np
import h5py

import logging
logger = logging.getLogger(__name__)

import spts.log
from spts.log import log_and_raise_error,log_warning,log_info,log_debug

import spts.denoiser
import spts.detect
import spts.analysis
import spts.threshold

class Worker:
    def __init__(self, conf, i0_offset=0, pipeline_mode=False, data_mount_prefix="", step_size=1):
        self.conf = conf
        self.data_mount_prefix = data_mount_prefix
        self.pipeline_mode = pipeline_mode
        self._i0_offset = i0_offset
        self._step_size = step_size
        self.i = None
        self.update()

    def _update_denoiser(self):
        if not hasattr(self, 'denoiser') or not is_same_dicts(self.conf["denoise"], self.denoiser.denoise_dict):
            method = self.conf["denoise"]["method"]
            if method == "gauss":
                self.denoiser = spts.denoiser.DenoiserGauss(sigma=self.conf["denoise"]["sigma"])
            elif method == "gauss2":
                self.denoiser = spts.denoiser.DenoiserGauss2(sigma=self.conf["denoise"]["sigma"])
            elif method == "histogram":
                images_bg = self._read_image(self.dataset_name, np.float64, self.conf["denoise"].get("n_histogram", 100))
                self.denoiser = spts.denoiser.DenoiserHistogram(window_size=self.conf["denoise"]["window_size"],
                                                               images_bg=images_bg,
                                                               vmin=self.conf["denoise"].get("vmin", -20),
                                                               vmax=self.conf["denoise"].get("vmax", 20),
                                                               dx=self.conf["denoise"].get("dx", 1),
                                                               vmin_full=self.conf["denoise"].get("vmin_full", -50),
                                                               vmax_full=self.conf["denoise"].get("vmax_full", 255))
            else:
                print "ERROR: Method %s is not implemented" % method
                return
            self.denoiser.denoise_dict = dict(self.conf["denoise"])

    def _is_valid_i(self, i):
        if (i is None) or (self.N_arr is None) or (self.N is None):
            return False
        else:
            return ((i < self.N_arr) and (i-self.conf["general"]["i0"] < self.N))
            
    def get_work(self):
        if self.pipeline_mode:
            print "ERROR: Do not use function get_work in pipeline mode!"
            return

        if self.i is None:
            i = self._i0_offset + self.conf["general"]["i0"]
        else:
            i = self.i + self._step_size
            
        if self._is_valid_i(i):
            self.i = i
            work_package = {"i": i}
            return work_package
        else:
            return None
            
    def work(self, work_package, tmp_package=None, out_package=None, target="analyse"):

        W = work_package

        # Update index
        i = W["i"]
        log_debug(logger, "(%i) Start work" % i)
        if self.pipeline_mode:
            self.update()

        if not self._is_valid_i(i):
            logger.warning("Invalid index. Probably we reached the end of the processing range (i=%i, N=%i, N_arr=%i)" % (i, self.N, self.N_arr))
            return None
        
        if tmp_package is None or W["i"] != tmp_package["i"]:
            tmp_package = {"i": i}
        else:
            out_package = dict(tmp_package)
        if out_package is None or W["i"] != out_package["i"]:
            out_package = {"i": i}
            
        tmp = [("1_raw", self._work_raw),
               ("2_process", self._work_process),
               ("3_denoise", self._work_denoise),
               ("4_threshold", self._work_threshold),
               ("5_detect", self._work_detect),
               ("6_analyse", self._work_analyse)]
           
        for work_name, work_func in tmp:
            if not work_name in tmp_package:
                log_debug(logger, "(%i) Starting %s" % (i, work_name))
                out_package, tmp_package = work_func(work_package, tmp_package, out_package)
                log_debug(logger, "(%i) Done with %s" % (i, work_name))
            if work_name.endswith(target):
                log_debug(logger, "(%i) Reached target %s" % (i, work_name))
                return out_package
        log_warning(logger, "(%i) Incorrect target defined (%s)" % (i, target))
        return out_package
        
    def _work_raw(self, work_package, tmp_package, out_package):
        i = work_package["i"]
        O = OutputCollector()
        # Read raw data
        image_raw, saturation_mask = self._load_data(i, self.conf["raw"]["dataset_name"], self.conf["raw"]["subtract_constant"], self.conf["raw"].get("cmcx", False), self.conf["raw"].get("cmcy", False), saturation_level=self.conf["raw"]["saturation_level"])
        O.add("image_raw", image_raw, 5, pipeline=True)
        O.add("saturation_mask", saturation_mask, 5, pipeline=True)
        # Measurement succeeded if all pixel values are below saturation level
        saturated_n_pixels = saturation_mask.sum()
        O.add("saturated_n_pixels", saturated_n_pixels, 0)
        success = (saturated_n_pixels == 0) or not self.conf["raw"]["skip_saturated_frames"]
        O.add("success", success, 0, pipeline=True)          
        out_package["1_raw"] = O.get_dict(self.conf["general"]["output_level"], self.pipeline_mode)
        tmp_package["1_raw"] = O.get_dict(5, True)
        return out_package, tmp_package
            
    def _work_process(self, work_package, tmp_package, out_package):
        i = work_package["i"]
        O = OutputCollector()
        # Read processed data
        image, foo = self._load_data(i, self.conf["process"]["dataset_name"], self.conf["process"]["subtract_constant"], self.conf["process"]["cmcx"], self.conf["process"]["cmcy"])
        if self.conf["process"]["floor_cut_level"] is not None:
            sel = image<self.conf["process"]["floor_cut_level"]
            if sel.sum() > 0:
                image[sel] = 0
        O.add("image", image, 2, pipeline=True)
        success = True
        O.add("success", success, 0, pipeline=True)        
        out_package["2_process"] = O.get_dict(self.conf["general"]["output_level"], self.pipeline_mode)
        tmp_package["2_process"] = O.get_dict(5, True)
        return out_package, tmp_package

    def _work_denoise(self, work_package, tmp_package, out_package):
        i = work_package["i"]
        O = OutputCollector()
        image = tmp_package["2_process"]["image"]
        # Denoise
        log_debug(logger, "(%i/%i) Denoise image" % (i+1, self.N_arr))
        self._update_denoiser()
        image_denoised = self.denoiser.denoise_image(image, full_output=True)
        O.add("image_denoised", np.asarray(image_denoised, dtype=np.int16), 4, pipeline=True)
        success = True
        O.add("success", success, 0, pipeline=True)        
        out_package["3_denoise"] = O.get_dict(self.conf["general"]["output_level"], self.pipeline_mode)
        tmp_package["3_denoise"] = O.get_dict(5, True)
        return out_package, tmp_package
        
    def _work_threshold(self, work_package, tmp_package, out_package):
        i = work_package["i"]
        O = OutputCollector()
        image_denoised = tmp_package["3_denoise"]["image_denoised"]
        image_thresholded = spts.threshold.threshold(image_denoised, self.conf["threshold"]["threshold"], fill_holes=self.conf["threshold"].get("fill_holes", False))
        O.add("image_thresholded", np.asarray(image_thresholded, dtype=np.bool), 3, pipeline=True)
        thresholded_n_pixels = image_thresholded.sum()
        O.add("thresholded_n_pixels", thresholded_n_pixels, 0)
        success = thresholded_n_pixels > 0
        O.add("success", success, 0, pipeline=True)        
        out_package["4_threshold"] = O.get_dict(self.conf["general"]["output_level"], self.pipeline_mode)
        tmp_package["4_threshold"] = O.get_dict(5, True)
        return out_package, tmp_package

    def _work_detect(self, work_package, tmp_package, out_package):
        i = work_package["i"]
        O = OutputCollector()
        image_denoised = tmp_package["3_denoise"]["image_denoised"]
        image_thresholded = tmp_package["4_threshold"]["image_thresholded"]
        # Detect particles
        log_debug(logger, "(%i/%i) Detect particles" % (i+1, self.N_arr))
        n_max = self.conf["detect"]["n_particles_max"]
        success, i_labels, image_labels, area, x, y, score, merged, dist_neighbor, dislocation = spts.detect.find_particles(image_denoised, image_thresholded,
                                                                                                                       self.conf["detect"]["min_dist"],
                                                                                                                       n_max,
                                                                                                                       peak_centering=self.conf["detect"]["peak_centering"])
        n_labels = len(i_labels)
        success = success and (n_labels > 0) and (n_labels <= n_max)
        log_info(logger, "(%i/%i) Found %i particles" % (i+1, self.N_arr, n_labels))
        if success:
            O.add("n", n_labels, 0, pipeline=True)
            O.add("x", uniform_particle_array(x, n_max), 0, pipeline=True)
            O.add("y", uniform_particle_array(y, n_max), 0, pipeline=True)
            O.add("peak_score", uniform_particle_array(score, n_max), 0, pipeline=False)
            O.add("area", uniform_particle_array(area, n_max, np.int32), 0)
            O.add("merged", uniform_particle_array(merged, n_max, np.int16), 0, pipeline=True)
            O.add("dist_neighbor", uniform_particle_array(dist_neighbor, n_max), 0)
            O.add("i_labels", uniform_particle_array(i_labels, n_max), 5, pipeline=True)
            O.add("image_labels", image_labels, 5, pipeline=True)
            O.add("dislocation", uniform_particle_array(dislocation, n_max), 0, pipeline=False)
        else:
            O.add("n", 0, 0, pipeline=True)
            O.add("x", uniform_particle_array([], n_max), 0, pipeline=True)
            O.add("y", uniform_particle_array([], n_max), 0, pipeline=True)
            O.add("peak_score", uniform_particle_array([], n_max), 0, pipeline=False)            
            O.add("area", uniform_particle_array([], n_max, np.int32), 0)
            O.add("merged", uniform_particle_array([], n_max, np.int16), 0, pipeline=True)
            O.add("dist_neighbor", uniform_particle_array([], n_max), 0)
            O.add("i_labels", uniform_particle_array([], n_max), 5, pipeline=True)
            O.add("image_labels", np.zeros_like(image_thresholded), 5, pipeline=True)
            O.add("dislocation", uniform_particle_array([], n_max), 0, pipeline=False)
        O.add("success", success, 0, pipeline=True)        
        out_package["5_detect"] = O.get_dict(self.conf["general"]["output_level"], self.pipeline_mode)
        tmp_package["5_detect"] = O.get_dict(5, True)
        return out_package, tmp_package

    def _work_analyse(self, work_package, tmp_package, out_package):
        i = work_package["i"]
        O = OutputCollector()
        image_raw = tmp_package["1_raw"]["image_raw"]
        saturation_mask = tmp_package["1_raw"]["saturation_mask"]
        image = tmp_package["2_process"]["image"]
        n_labels = tmp_package["5_detect"]["n"]
        i_labels = tmp_package["5_detect"]["i_labels"]
        i_labels = i_labels[i_labels != -1]
        image_labels = tmp_package["5_detect"]["image_labels"]
        x = tmp_package["5_detect"]["x"]
        x = x[x != -1]
        y = tmp_package["5_detect"]["y"]
        y = y[y != -1]
        merged = tmp_package["5_detect"]["merged"]
        n_max = self.conf["detect"]["n_particles_max"]
        res = spts.analysis.analyse_particles(image=image,
                                             image_raw=image_raw,
                                             saturation_mask=saturation_mask,
                                             i_labels=i_labels,
                                             labels=image_labels,
                                             x=x, y=y,
                                             merged=merged,
                                             full_output=self.conf["general"]["output_level"] >= 3,
                                             n_particles_max=n_max,
                                             **self.conf["analyse"])
        success, peak_success, peak_sum, peak_mean, peak_median, peak_min, peak_max, peak_size, peak_saturated, peak_eccentricity, peak_circumference, masked_image, peak_thumbnails = res
        # Analyse image at particle positions
        log_debug(logger, "(%i/%i) Analyse image at %i particle positions" % (i, self.N_arr, len(i_labels)))
        #if n_labels > self.n_particles_max:
        #    log_warning(logger, "(%i/%i) Too many particles (%i/%i) - skipping analysis for %i particles" % (i_image+1, self.N_arr, n_labels, self.n_particles_max, n_labels - self.n_particles_max))
        O.add("peak_success", uniform_particle_array(peak_sum, n_max, np.bool, vinit=False), 0)
        O.add("peak_sum", uniform_particle_array(peak_sum, n_max), 0, pipeline=True)
        O.add("peak_mean", uniform_particle_array(peak_mean, n_max), 0)
        O.add("peak_median", uniform_particle_array(peak_median, n_max), 0)
        O.add("peak_min", uniform_particle_array(peak_min, n_max), 0)
        O.add("peak_max", uniform_particle_array(peak_max, n_max), 0)
        O.add("peak_size", uniform_particle_array(peak_size, n_max), 0)
        O.add("peak_eccentricity", uniform_particle_array(peak_eccentricity, n_max), 0, pipeline=True)
        O.add("peak_circumference", uniform_particle_array(peak_circumference, n_max), 0)
        O.add("peak_saturated", uniform_particle_array(peak_saturated, n_max, np.int8, vinit=0), 0)
        if success:
            if self.conf["analyse"]["integration_mode"] == "windows":
                s = self.conf["analyse"]["window_size"]
            else:
                s = spts.analysis.THUMBNAILS_WINDOW_SIZE_DEFAULT
        if peak_thumbnails is not None and success:
            O.add("peak_thumbnails", np.asarray(peak_thumbnails, dtype=np.int32), 3)
        else:
            O.add("peak_thumbnails", np.zeros(shape=(n_max, s, s), dtype=np.int32), 3)
        if masked_image is not None and success:
            O.add("masked_image", np.asarray(masked_image, dtype=np.int32), 3, pipeline=True)            
        else:
            O.add("masked_image", np.zeros(shape=image.shape, dtype=np.int32), 3, pipeline=True)
        O.add("success", success, 0, pipeline=True)
        out_package["6_analyse"] = O.get_dict(self.conf["general"]["output_level"], self.pipeline_mode)
        tmp_package["6_analyse"] = O.get_dict(5, True)
        return out_package, tmp_package

    def _load_data(self, i, dataset_name, subtract_constant, xcmc, ycmc, saturation_level=None):
        # Load data from file at index "i"
        image = self._read_image(i, dataset_name, np.int32)
        if saturation_level is not None:
            saturation_mask = image >= saturation_level
        else:
            saturation_mask = None
        if subtract_constant is not None:
            image -= subtract_constant
        if xcmc:
            med = np.median(image, axis=1)
            med_image = np.repeat(med, image.shape[1]).reshape(image.shape[0], image.shape[1])
            image -= np.asarray(med_image, dtype=image.dtype)
        if ycmc:
            med = np.median(image, axis=0)
            med_image = np.repeat(np.asarray([med]), image.shape[0], axis=0).reshape(image.shape[0], image.shape[1])
            image -= np.asarray(med_image, dtype=image.dtype)
        return image, saturation_mask

    def _get_full_filename(self):
        if len(self.data_mount_prefix) == 0:
            fn = self.conf["general"]["filename"]
        else:
            fn = "%s/%s" % (self.data_mount_prefix, self.conf["general"]["filename"])
        return fn
        
    def _read_image(self, i, dataset_name, dtype, N=1):
        fn = self._get_full_filename()
        with h5py.File(fn, "r") as f:
            if dataset_name not in f:
                raise IOError("Cannot find dataset %s in %s." % (dataset_name, fn))
            xmin = self.conf["raw"]["xmin"] if self.conf["raw"]["xmin"] is not None else 0
            xmax = self.conf["raw"]["xmax"] if self.conf["raw"]["xmax"] is not None else Nx
            ymin = self.conf["raw"]["ymin"] if self.conf["raw"]["ymin"] is not None else 0
            ymax = self.conf["raw"]["ymax"] if self.conf["raw"]["ymax"] is not None else Ny
            if N == 1:
                return np.asarray(f[dataset_name][i,ymin:ymax,xmin:xmax], dtype=dtype)
            else:
                return np.asarray(f[dataset_name][i:i+N,ymin:ymax,xmin:xmax], dtype=dtype)

    def update(self):
        fn = self._get_full_filename()
        with h5py.File(fn, "r") as f:
            assert f[self.conf["raw"]["dataset_name"]].shape[0] == f[self.conf["process"]["dataset_name"]].shape[0]
            self.N_arr = f[self.conf["raw"]["dataset_name"]].shape[0]    
        if self.conf["general"]["n_images"] is None or self.conf["general"]["n_images"] > 0:
            self.N = self.N_arr
        else:
            self.N = self.conf["general"]["n_images"]
            

def uniform_particle_array(v, n_max, dtype=np.float64, vinit=-1):
    n = min([n_max,len(v)])
    A = np.zeros(n_max, dtype=dtype)
    A[:] = vinit
    for i in range(n):
        if v[i] is not None:
            A[i] = v[i]
    return A


class OutputCollector:
    
    def __init__(self):
        self._D = {}
        self._D_output_level = {}
        self._D_pipeline = {}

    def add(self, name, item, output_level=0, pipeline=False):
        self._D[name] = item
        self._D_output_level[name] = output_level
        self._D_pipeline[name] = pipeline
        
    def exists(self, name):
        return name in self._D
        
    def get(self, name):
        return self._D[name]
        
    def get_dict(self, output_level=0, pipeline=False):
        O = {}
        for k in self._D_output_level.keys():
            l = self._D_output_level[k]
            p = self._D_pipeline[k]
            if l <= output_level or (p and pipeline):
                O[k] = self._D[k]
        return O
    

def is_same_dicts(d1, d2):
    same = len(d1) == len(d2)
    same *= len(d1) == len(set(d1.items()) & set(d2.items()))
    return same
