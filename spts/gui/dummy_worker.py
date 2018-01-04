import numpy as np


class DummyWorker:

    def __init__(self, conf):
        self.N = 1
        self.conf = conf

    def work(self, work_package, tmp_package=None, out_package=None, target="analyse"):
        dummy_image = 100*np.random.rand(200, 100)
        res = {
            "1_raw" : {"image_raw" :  dummy_image},
            "2_process" : {"image" : dummy_image},
            "3_denoise" : {"image_denoised" : dummy_image},
            "4_threshold" : {"image_thresholded" : dummy_image},
            "5_detect" : {"image_labels" : dummy_image,
                          "x" : np.array([-1]),
                          "y" : np.array([-1]),},
            "6_analyse" : {"masked_image" : dummy_image,
                           "peak_sum" : np.array([-1]),
                           "peak_eccentricity" : np.array([-1]),},
        }
        return res

    

    
