from distutils.core import setup, Extension
import numpy.distutils.misc_util 

ext = Extension("denoise", sources=["denoise_module.c"])
setup(name="denoise",ext_modules=[ext], include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs())
