import numpy
from scipy.misc import derivative

import fj as fj_c
import msi.utils.free_jet as fj_py

d_p = 100E-9
z = 2E-3
#D = 1E-3
D = 0.001
#T0 = 300.
T0 = 300.
#p0 = 100.
p0 = 0.5*100
#v_p_0 = 100.
v_p_0 = 30.
z_p_max = 20E-3

# C calculations
fj_c.test(p0=p0, T0=T0, D=D, d_p=d_p, v_p=v_p_0, z_p=z, gas=1)

# Python calculations
fj_py.p0 = p0
fj_py.T0 = T0
fj_py.gas = "He"

M_c_t = fj_py.M_c(z, D)
print "M_c = %g" % M_c_t
Re_t = fj_py.Re(v_p_0, z, D, d_p)
print "Re = %g" % Re_t
print "rho0 = %g" % fj_py.rho0()
print "eta = %g" % fj_py.eta(T0)
T_c_t = fj_py.T_c(z, D)
print "T_c = %g" % T_c_t
print "p_c = %g" % fj_py.p_c(z, D)
print "rho_c = %g" % fj_py.rho_c(z, D)
print "v_g_c = %g" % fj_py.v_g_c(z, D)
print "C_D_c = %g" % fj_py.C_D_c(v_p_0, z, D, d_p)
print "C_D_supersonic = %g" % fj_py.C_D_supersonic(Re_t, fj_py.T_p, T_c_t, M_c_t, fj_py.S(M_c_t))
print "C_D_subsonic = %g" % fj_py.C_D_subsonic(Re_t, fj_py.T_p, T_c_t, M_c_t, fj_py.S(M_c_t))
print "dvdt_p_c = %g" % fj_py.dv_p_c(v_p_0, z, D, d_p)
print "ddvdtdz_p_c = %g" % derivative(lambda z: fj_py.dv_p_c(v_p_0, z, D, d_p), z, dx=1E-6)
print "ddvddt_p_c = %g" % fj_py.ddv_p_c(v_p_0, z, D, d_p)


if True:

    import cProfile
    cProfile.run('z_c, v_c, a_c = fj_c.iterate_particle_motion(p0=p0, T0=T0, gas=1, D=D, d=d_p, v_0=v_p_0, z_max=z_p_max)')
    cProfile.run('z_py, v_py, a_py = fj_py.iterate_particle_motion(D=D, d_p=d_p, v_p_0=v_p_0, z_p_0=0., z_p_max=z_p_max)')

    print z_c, z_py
    print v_c, v_py

    import matplotlib.pyplot as pypl
    pypl.figure()
    pypl.plot(z_c, v_c)
    pypl.plot(z_py, v_py)
    pypl.legend(["C", "Python"])
    pypl.ylim(0, None)
    pypl.show()
