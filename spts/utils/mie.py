import numpy as np
from bhmie import bhmie

wavelength = 532.E-9
NA = 0.055

# Refractive index
# polystyrene: n = 1.5916
# source: http://refractiveindex.info/?shelf=organic&book=polystyren&page=Sultanova
n_ps = 1.5983
# sucrose: n = 1.5376
# Source: https://pubchem.ncbi.nlm.nih.gov/compound/sucrose#section=Crystal-Structures
# (Weast, R.C. (ed.). Handbook of Chemistry and Physics. 60th ed. Boca Raton, Florida: CRC Press Inc., 1979., p. C-503)
n_suc = 1.5376
# water
n_wat = 1.3325
# cytosol
n_cyt = 1.375
# mitochondria
n_mit = 1.41

# Convert Rayleigh diameter (ph/mJ)^(1/6) to particle diameter in nm
ray_to_nm = {
    "201612": 31.04,
    "201605": 33.07,
}
ray2nm = lambda ray, calib, n=n_ps: ray * ray_to_nm[calib] * ( ((n_ps**2-1)/(n_ps**2+2))**(1/3.) ) / ( ((n**2-1)/(n**2+2))**(1/3.) )

# choosing material
n = n_ps

I_pulse = 1.
R = 1.
A = 1.

# Rayleigh scattering
# ===================

# Polarisation factor for in-plane a_off
_P = lambda a_off: np.cos(a_off)**2
# Source: https://en.wikipedia.org/wiki/Rayleigh_scattering
_I_Ray = lambda diameter, a_off=0.: (I_pulse * A * _P(a_off) / R**2 * (2*np.pi/wavelength)**4 * ((n**2-1)/(n**2+2))**2 * (diameter/2.)**6)
I_Ray = lambda diameter, a_off=0.: _I_Ray(diameter, a_off)

# Mie scattering
# ==============

# Size parameter
_size_parameter = lambda diameter: 2*np.pi/wavelength * diameter/2.
# Intensity (assuming polarisation fully perpendicular to scattering plane, S2=0)
_i_S2 = 1
# The factor "wavelength/(2*pi))**2" is guessed by comparing with Rayleigh scattering
I_Mie = lambda diameter, a_in: (wavelength/(2*np.pi))**2 * I_pulse * A / R**2 * abs((bhmie(_size_parameter(diameter), n, [a_in])[_i_S2]))[0]**2

