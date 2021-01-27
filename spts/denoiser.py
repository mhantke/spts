import numpy as np
#import scipy.ndimage       

class DenoiserGauss:

    def __init__(self, sigma):
        self.sigma = sigma
        self._G = None
        self._G_sigma = None

    def denoise_image(self, image, full_output=False):
        image_f64 = np.asarray(image, dtype=np.float64)
        # scipy (scipy defines sigma here weirdly ... so I better use my own gaussian)
        #image_smoothed = scipy.ndimage.gaussian_filter(image_f64, self.window_size/2.)
        if self._G is None or self._G_sigma != self.sigma:
            Nx = image.shape[1]
            Ny = image.shape[0]
            X,Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
            X = np.float64(X - Nx/2) / Nx
            Y = np.float64(Y - Ny/2) / Ny
            Rsq = X**2 + Y**2
            self._G = np.exp(-Rsq/2./self.sigma**2)
            self._G_sigma = self.sigma
        fimage_G = np.fft.fftshift(np.fft.fft2(image)) * self._G
        image_G = np.fft.ifft2(np.fft.ifftshift(fimage_G))
        return image_G.real

class DenoiserGauss2: 

    def __init__(self, sigma): 
        self.sigma1 = sigma*2
        self.sigma2 = sigma
        self._G1 = None
        self._G2 = None
        self._G1_sigma = None
        self._G2_sigma = None

    def denoise_image(self, image, full_output=False):
        image_f64 = np.asarray(image, dtype=np.float64)
        Nx = image.shape[1]
        Ny = image.shape[0]
        if self._G1 is None or self._G1_sigma != self.sigma1:
            X,Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
            X = np.float64(X - Nx/2) / Nx
            Y = np.float64(Y - Ny/2) / Ny
            Rsq = X**2 + Y**2
            self._G1 = np.exp(-Rsq/2./(self.sigma1+np.finfo(np.float64).eps)**2)
            self._G1_sigma = self.sigma1
        if self._G2 is None or self._G2_sigma != self.sigma2:
            X,Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
            X = np.float64(X - Nx/2) / Nx
            Y = np.float64(Y - Ny/2) / Ny
            Rsq = X**2 + Y**2
            self._G2 = np.exp(-Rsq/2./(self.sigma2+np.finfo(np.float64).eps)**2)
            self._G2_sigma = self.sigma2
        fimage_G1 = np.fft.fftshift(np.fft.fft2(image)) * self._G1
        image_G1 = np.fft.ifft2(np.fft.ifftshift(fimage_G1))
        fimage_G2 = np.fft.fftshift(np.fft.fft2(image)) * self._G2
        image_G2 = np.fft.ifft2(np.fft.ifftshift(fimage_G2))
        out = (abs(image_G1) - abs(image_G2))
        #print out.max(), out.min()
        #out = out/out.max() * 65000.
        return out

class DenoiserHistogram:

    def __init__(self, window_size, images_bg, vmin, vmax, dx=1, vmin_full=-50, vmax_full=255):
        import spts.denoise
        images_bg_i32 = np.asarray(images_bg, dtype=np.int32) # No copy is performed with asarray if images_bg has right dtype
        self.window_size = window_size
        self.vmin = vmin
        self.vmax = vmax
        self.dx = dx
        self.vmin_full = vmin_full
        self.vmax_full = vmax_full
        self.hists_bg = spts.denoise.calc_hists(images_bg_i32, vmin=vmin, vmax=vmax, window_size=window_size, dx=self.dx)

    def denoise_image(self, image, full_output=False):
        image_i32 = np.asarray(image, dtype=np.int32)
        image_scores = denoise_hist.denoise(image_i32, self.hists_bg, vmin=self.vmin, vmax=self.vmax, window_size=self.window_size, vmin_full=self.vmin_full, vmax_full=self.vmax_full, dx=self.dx)
        return image_scores
