#!/usr/bin/env python

#--------------------------------------------------------------------------
# SPTS - Single Particle Tracking and Sizing
# Copyright 2018 Max Felix Hantke
#--------------------------------------------------------------------------

import sys, os
this_dir = os.path.dirname(os.path.realpath(__file__))
print(this_dir)
sys.path.append(this_dir + "/spts/data")
#from distutils.core import setup, Extension
from setuptools import setup, Extension, find_packages

import numpy
os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'
setup(    
    name='spts',
    version='0.0.2',
    description='SPTS',
    long_description='SPTS - Single Particle Tracking and Sizing',
    #url='',
    author='Hantke, Max Felix',
    author_email='hantke@xray.bmc.uu.se',
    license='BSD',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        #'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.2',
        #'Programming Language :: Python :: 3.3',
        #'Programming Language :: Python :: 3.4',
    ],

    keywords='mie scattering single particle',

    packages = find_packages(),

    package_data={
        '': [
            'gui/ui/*.ui',
            'gui/spts_default.conf'
        ]
    },


    install_requires=['numpy', 'scipy', 'h5py', 'h5writer', 'mulpro>=0.1.3'],

    extras_require={'mpi': 'mpi4py>=1.3.1',
                    'gui': ['PyQt4', 'pyqtgraph']},
   
    ext_modules=[
        Extension(
            "spts.denoise",
            sources=["spts/denoise_module.cpp"],
            include_dirs=[numpy.get_include()],
            ),
        Extension(
            "spts.utils.fj",
            sources=["spts/utils/fj/fj_module.cpp"],
            include_dirs=[numpy.get_include()],
            ),
        ],
    
    scripts=[this_dir+"/spts/scripts/"+s for s in os.listdir(this_dir+"/spts/scripts/") if ((s.endswith(".py") or s.endswith(".sh")) and not (s.startswith(".")))],
)
