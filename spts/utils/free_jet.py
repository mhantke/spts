
import scipy.constants
from scipy.misc import derivative
from scipy.special import erf
import numpy as np

#############
# Constants #
#############

gas = "He"
gases = ["air", "He"]

# Molar mass
M_mol = {"air": 28.97E-3, "He": 4.002602E-3}
# Gas molecule mass
m_g = lambda: M_mol[gas] / scipy.constants.N_A

# Ratio of specific heats cp/cv
gamma = {"air": 1.4, "He": 1.67, "CO2": 9./7.}

# Collision molecule diameter
d_mol = {"air": 4.0E-10, "He": 2.60E-10}
sigma_mol = {"air": np.pi*(d_mol["air"]/2.)**2,
             "He": np.pi*(d_mol["He"]/2.)**2}

# Stagnation state
T0 = 293. # Rom temperature
p0 = 0.5 * 100 # Pressure at outlet of the injector in Pa (SI) (1 mbar = 100 Pa)

# Particle mass density
rho_p = 1050. # PS spheres
# Particle mass
m_p = lambda d_p: (4./3. * np.pi * (d_p/2.)**3) * rho_p

# Reduced mass of collision (Schwartz & Andres, 1976)
m_red = lambda d_p: (m_g() * m_p(d_p)) / (m_g() + m_p(d_p))

##########################
# Centerline Mach number #
##########################

# Spatial scaling with effective nozzle diameter D
_x = lambda z, D: z/D 

# Close regime (0 < x < 2)
_Ap = {"air": 1.6515}
_Bp = {"air": 0.1164}
# Values for He are defined below
_M_c1 = lambda x: 1.0 + _Ap[gas]*x - _Bp[gas] * x**2
M_c1 = lambda z, D: _M_c1(_x(z, D))

# Far regime (2 < x < 6)
_A = {"air": 3.65, "He": 3.26, "CO2": 3.96} # Ashkenas and Sherman 1966
_x0 = {"air": 0.40, "He": 0.075, "CO2": 0.85}
_B = lambda x: abs(x-_x0[gas])**(gamma[gas]-1)
_C = {"air": 0.20, "He": 0.31, "CO2": 0.} # 0 means nothing fits here according to Ashkenas and Sherman (1966)
_M_c2 = lambda x: _A[gas]*_B(x) - \
                      0.5 * (gamma[gas]+1) / (gamma[gas]-1) / (_A[gas]*_B(x)) + \
                      _C[gas] / _B(x)**3
M_c2 = lambda z, D: _M_c2(_x(z, D))
M_c = lambda z, D: np.where(_x(z, D) < 2, M_c1(z, D), M_c2(z, D))

# Estimate close regime for He
#tmp = gas
#gas = "He"
#M2 = M_c2(2., 1.)
#dM2 = derivative(lambda z: M_c2(z, 1.), 2.)
#_Ap["He"] = M2 - dM2 - 1
#_Bp["He"] = -(dM2 - _Ap["He"])/4.
#gas = tmp
# These are the values that come out:
_Ap["He"] = 1.9354952074905416
_Bp["He"] = 0.034763777474581481

########################################
# Local state parameters at centerline #
########################################

# Stagnation state
rho0 = lambda: p0 * M_mol[gas] / (scipy.constants.R * T0) # Ideal gas

tmp = gas
gas = "air"
assert np.isclose(rho0()/p0, 1.2041/scipy.constants.atmosphere)
gas = "He"
assert np.isclose(rho0()/p0, 0.1664/scipy.constants.atmosphere)
gas = tmp

# Local state on central line
T_c = lambda z, D: T0 / (1+(gamma[gas]-1)*M_c(z, D)**2/2.)
p_c = lambda z, D: p0 / (1+(gamma[gas]-1)*M_c(z, D)**2/2.)**(gamma[gas]*(gamma[gas]-1))
rho_c = lambda z, D: rho0() / (1+(gamma[gas]-1)*M_c(z, D)**2/2.)**(1./(gamma[gas]-1))

# Local number density of gas molecules
n_c = lambda z, D: rho_c(z, D) / m_g()

#################
# Gas molecules #
#################

# Gas velocity
v_g_c = lambda z, D: M_c(z, D) * np.sqrt(gamma[gas]*scipy.constants.R*T_c(z, D)/M_mol[gas])

# Mean free path
mfp_c = lambda z, D: scipy.constants.k * T_c(z, D) / (np.sqrt(2) * np.pi * d_mol[gas]**2 * p_c(z, D))

# Dynamic gas viscosity (ideal gas)
# Wikipedia (https://en.wikipedia.org/wiki/Viscosity)
T_eta = {"air": 291.15, "He": 273.}
eta_eta = {"air": 18.27E-6, "He": 19E-6}
C_eta = {"air": 120., "He": 79.4}
eta = lambda T: eta_eta[gas] * (T_eta[gas] + C_eta[gas])/(T + C_eta[gas]) * (T/T_eta[gas])**(3/2.)

# Kinematic gas viscosity
nu = lambda T, rho: eta(T) * rho
nu_c = lambda z, D: eta(T_c(z, D)) * rho_c(z, D)

#################
# Particle drag #
#################

# Knudsen number
Kn = lambda z, D, d_p: mfp_c(z, D)/(d_p/2.)

# Sutherland's law
# Dynamic viskosity (often also mu) (https://en.wikipedia.org/wiki/Viscosity)
T_eta = {"air": 291.15, "He": 273.}
eta_eta = {"air": 18.27E-6, "He": 19E-6}
C_eta = {"air": 120., "He": 79.4}
eta = lambda T: eta_eta[gas] * (T_eta[gas] + C_eta[gas])/(T + C_eta[gas]) *(T/T_eta[gas])**(3/2.)

# Kinematic viscosity
nu = lambda T, rho: eta(T) / rho
nu_c = lambda z, D: eta(T_c(z, D)) / rho_c(z, D)

# Reynolds number
Re = lambda v_p, z, D, d_p: d_p*rho_c(z, D)*abs(v_p-v_g_c(z, D))/eta(T_c(z, D))

# Molecular speed ratio (Hendersen 1976)
S = lambda M: M * np.sqrt(gamma[gas] / 2.)

# Speed of sound (https://en.wikipedia.org/wiki/Speed_of_sound)
c_c = lambda z, D: np.sqrt( gamma[gas] * p_c(z, D) / rho_c(z, D) )

# Stokes number (Wang & McMurphy, 2006)
#Stk_c = lambda z, D: c_c(z, D) / D

# Mach number based on velocity difference between particle and gas
delta_v_c = lambda z, D, v_p: v_p - v_g_c(z, D)
M_p = lambda z, D, v_p: abs(delta_v_c(z, D, v_p)) / c_c(z, D)

# Drag coefficient for all flow conditions (Hendersen 1976)
# M < 1
C_D_subsonic = lambda Re, T_p, T_g, M, S: 24. / ( Re + S * ( 4.33 + ( 3.65 - 1.53 * T_p / T_g) / ( 1.0 + 0.353 * T_p / T_g ) * np.exp( -0.247 * Re / S ) ) ) \
               + ( np.exp( -0.5 * M / np.sqrt(Re) ) * \
                   ( ( 4.5 + 0.38 * ( 0.03 * Re + 0.48 * np.sqrt(Re) ) ) / ( 1.0 + 0.03 * Re + 0.48 * np.sqrt(Re) ) \
                     + 0.1 * M**2 \
                     + 0.2 * M**8 ) ) \
               + ( 1 - np.exp( - M / Re ) ) * 0.6 * S
# M > 1.75
C_D_supersonic = lambda Re, T_p, T_g, M, S: ( 0.9 \
                                           + 0.34 / M**2 \
                                           + 1.86 * np.sqrt( M / Re ) * ( 2. + 2. / S**2 + 1.058 / S * np.sqrt( T_p / T_g ) - 1/S**4 ) ) \
                                           / ( 1. + 1.86 * np.sqrt( M / Re ) )

_C_D_extreme = lambda Re, T_p, T_g, M: np.where((M <= 1), C_D_subsonic(Re, T_p, T_g, M, S(M)), C_D_supersonic(Re, T_p, T_g, M, S(M)))

_C_D_interp = lambda Re, T_p, T_g, M:  C_D_subsonic(Re, T_p, T_g, 1., S(1.)) \
              + 4/3. * ( M - 1. ) * ( C_D_supersonic(Re, T_p, T_g, 1.75, S(1.75)) -  C_D_subsonic(Re, T_p, T_g, 1., S(1.)) )

# M_infinity ~ M for Helium free jet at all distances, therefore I do not distinguish here
C_D = lambda Re, T_p, T_g, M: np.where((M > 1)*(M < 1.75), _C_D_interp(Re, T_p, T_g, M), _C_D_extreme(Re, T_p, T_g, M))

T_p = 293.
C_D_c = lambda v_p, z, D, d_p: C_D(Re(v_p, z, D, d_p), T_p, T_c(z, D), M_c(z, D))

# Particle acceleration
dv_p_c = lambda v_p, z, D, d_p: C_D_c(v_p, z, D, d_p) * rho_c(z, D) * delta_v_c(z, D, v_p)**2 * np.pi * (d_p/2.)**2 / 2. / m_p(d_p)
# First derivative of particle acceleration
ddv_p_c = lambda v_p, z, D, d_p: v_p * derivative(lambda z: dv_p_c(v_p, z, D, d_p), z, dx=1E-6)

def iterate_particle_motion(D, d_p, v_p_0=10., z_p_0= 0., z_p_max=1E-3, g=0.002, ddt_max=0.1, dt_min=1E-6):
    # g: Numerical parameter (iteration step size scaling constant)
    N = 10000
    z_p = [np.zeros(N)]
    v_p = [np.zeros(N)]
    a_p = [np.zeros(N)]
    z_p[0][0] = z_p_0
    v_p[0][0] = v_p_0
    a_p[0][0] = ddv_p_c(v_p_0, z_p_0, D, d_p)
    dt = 1000
    i = 1
    k = 1
    while z_p_max > z_p[-1][k-1]:
        k = i % N
        if k == 0:
            z_p.append(np.zeros(N))
            v_p.append(np.zeros(N))
            a_p.append(np.zeros(N))
        tmp_dv_p = dv_p_c(v_p[-1][k-1], z_p[-1][k-1], D, d_p)
        tmp_ddv_p = ddv_p_c(v_p[-1][k-1], z_p[-1][k-1], D, d_p)
        dt = 2 * g * abs(tmp_dv_p) / abs(tmp_ddv_p)
        #print dt
        #dt = min([dt, (1+ddt_max)*dt])
        #dt = max([dt, dt_min])
        a_p[-1][k] = a_p[-1][k-1] + dt * tmp_ddv_p
        v_p[-1][k] = v_p[-1][k-1] + dt * tmp_dv_p + 0.5 * dt**2 * tmp_ddv_p
        z_p[-1][k] = z_p[-1][k-1] + dt * v_p[-1][k]
        i += 1
    z_p = np.asarray(z_p).flatten()[:i]
    v_p = np.asarray(v_p).flatten()[:i]
    a_p = np.asarray(a_p).flatten()[:i]
    return z_p, v_p, a_p

def zva(params, i_params, pressure, size, dt_min=1E-5, T0=293., z_max=20E-3, gas="He", v_0=None):
    import spts.utils.fj as fj_c
    return fj_c.iterate_particle_motion(p0=params[i_params["p0 factor"]]*(pressure*100),
                                        T0=T0,
                                        gas=gases.index(gas),
                                        D=params[i_params["D"]],
                                        d=size*1E-9,
                                        v_0=v_0 if v_0 is not None else params[i_params["v_p_0"][pressure][size]],
                                        z_max=z_max)

def v(params, i_params, pressure, size, z_exp):
    z,v,a = zva(params, i_params, pressure, size)
    return np.interp(z_exp, z, v)


