#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>

int i;

// Physical constants
double N_A = 6.022140857e+23;
double R = 8.3144598;
double k_B = 1.38064852e-23;

// GAS PROPERTIES
// --------------

#define NGAS 2
//static int const GAS_AIR = 0;
//static int const GAS_HE = 1;

// Molar mass
static double M_g[NGAS] = { 28.97E-3, 4.002602E-3 };

// Ratio of specific heats cp/cv
static double gamma_g[NGAS] = { 1.40, 1.67 };

// Collision molecule diameter
static double d_coll_g[NGAS] = { 4.0E-10, 2.6E-10 };

// Stagnation state
// ...

// Kinematic gas viscosity (ideal gas)
// Wikipedia (https://en.wikipedia.org/wiki/Viscosity)
static double T_eta[NGAS] = { 291.15, 273. };
static double eta_eta[NGAS] = { 18.27E-6, 19E-6 };
static double C_eta[NGAS] = { 120., 79.4 };
double eta(double T, int gas) {
  return eta_eta[gas] * (T_eta[gas] + C_eta[gas]) / (T + C_eta[gas]) * pow(T/T_eta[gas],1.5);
}

// CENTERLINE MACH NUMBER
//-----------------------

// Spatial scaling with effective nozzle diameter D
double z_to_x(double z, double D) {
  return z/D;
}

// Close regime (0 < x < 2)
static double A_close[NGAS] = { 1.6515, 1.9354952074905416};
static double B_close[NGAS] = { 0.1164, 0.034763777474581481};
double M_c_close(double x, int gas) {
  return 1.0 + A_close[gas] * x - B_close[gas] * x*x;
}

// Far regime (2 < x < 6)
static double A_far[NGAS] = { 3.65, 3.26 }; // Ashkenas and Sherman 1966
static double x0_far[NGAS] = { 0.40, 0.075 };
double B_far(double x, int gas) {
  return pow(fabs(x-x0_far[gas]), gamma_g[gas]-1);
}
static double C_far[NGAS] = { 0.20, 0.31 };
double M_c_far(double x, int gas) {
  double _B_far = B_far(x, gas);
  return A_far[gas]*_B_far - 0.5 * (gamma_g[gas]+1) / (gamma_g[gas]-1) / (A_far[gas]*_B_far) + C_far[gas] / pow(_B_far, 3.);
}

double M_c(double x, int gas) {
  if (x < 2) {
    return M_c_close(x, gas);
  } else {
    return M_c_far(x, gas);
  }
}

// LOCAL STATE PARAMETERS
//-----------------------

double T_c(double x, int gas, double T0) {
  double _M_c = M_c(x, gas);
  return T0 / (1 + (gamma_g[gas]-1)*(_M_c*_M_c)/2.);
}
double p_c(double x, int gas, double p0) {
  double _M_c = M_c(x, gas);
  return p0 / pow(1 + (gamma_g[gas]-1)*(_M_c*_M_c)/2., gamma_g[gas]*(gamma_g[gas]-1));
}
// Ideal gas law
double rho0(int gas, double p0, double T0) {
  return p0 * M_g[gas] / (R * T0);
}
double rho_c(double x, int gas, double p0, double T0) {
  double _M_c = M_c(x, gas);
  double _rho0 = rho0(gas, p0, T0);
  return _rho0 / pow(1 + (gamma_g[gas]-1)*_M_c*_M_c/2.,1./(gamma_g[gas]-1.));
}

// GAS VELOCITY
//-------------
double v_g_c(double x, int gas, double T0) {
  return M_c(x, gas) * sqrt(gamma_g[gas]*R*T_c(x, gas, T0)/M_g[gas]);
}

// MEAN FREE PATH
//---------------
double mfp_c(double x, int gas, double p0, double T0) {
  return k_B*T_c(x, gas, T0)/sqrt(2.)*M_PI*(d_coll_g[gas]*d_coll_g[gas])*p_c(x, gas, p0);
}

// PARTICLE PROPERTIES
//--------------------

// Particle temperature (assumed to be constant, probably bad assumption but no better easily at hand)
double T_p = 293.;
// Particle mass density
double rho_p = 1050.; // Polystyrene spheres that we use
// Particle mass
double m_p(double d_p) {
  return rho_p*4./3.*M_PI*pow(d_p/2., 3.);
}

// PARTICLE DRAG
//--------------

// Knudsen number
double Kn(double x, int gas, double d_p, double p0, double T0) {
  return mfp_c(x, gas, p0, T0)/(d_p/2.);
}

// Reynolds number
double Re(double x, int gas, double d_p, double v_p, double p0, double T0) {
  return d_p*rho_c(x, gas, p0, T0)*fabs(v_p-v_g_c(x, gas, T0))/eta(T_c(x, gas, T0), gas);
}

// Molecular speed ratio (Hendersen 1976)
double S(double M, int gas) {
  return M * sqrt(gamma_g[gas] / 2.);
}

// Drag coefficient for all flow conditions (Hendersen 1976)
// M < 1
double C_D_subsonic(double Re, double T_p, double T_g, double M, double S) {
  return 24. / ( Re + S * ( 4.33 + ( 3.65 - 1.53 * T_p / T_g) / ( 1.0 + 0.353 * T_p / T_g ) * exp( -0.247 * Re / S ) ) )
    + ( exp( -0.5 * M / sqrt(Re) ) *
	( ( 4.5 + 0.38 * ( 0.03 * Re + 0.48 * sqrt(Re) ) ) / ( 1.0 + 0.03 * Re + 0.48 * sqrt(Re) ) 
	  + 0.1 * M*M 
	  + 0.2 * pow(M, 8.) ) ) 
    + ( 1 - exp( - M / Re ) ) * 0.6 * S;
}
// M > 1.75
double C_D_supersonic(double Re, double T_p, double T_g, double M, double S) {
  return ( 0.9							 
	   + 0.34 / (M*M)
           + 1.86 * sqrt( M / Re ) * ( 2. + 2. / (S*S) + 1.058 / S * sqrt( T_p / T_g ) - pow(S, -4.) ) ) / ( 1. + 1.86 * sqrt( M / Re ) );
}
// Interpolation in transition conditions
double _C_D_interp(double Re, double T_p, double T_g, double M, int gas) {
  return C_D_subsonic(Re, T_p, T_g, 1., S(1., gas)) + 4/3. * ( M - 1. ) * ( C_D_supersonic(Re, T_p, T_g, 1.75, S(1.75, gas)) -  C_D_subsonic(Re, T_p, T_g, 1., S(1., gas)) );
}
// Combine
double C_D(double Re, double T_p, double T_g, double M, int gas) {
  if ((M > 1.)&&(M < 1.75)) {
    return _C_D_interp(Re, T_p, T_g, M, gas);
  } else {
      if (M <= 1.) {
	return C_D_subsonic(Re, T_p, T_g, M, S(M, gas));
      } else {
	return C_D_supersonic(Re, T_p, T_g, M, S(M, gas));
      }
  }
}
// Fot the centerline
double C_D_c(double x, int gas, double d_p, double v_p, double p0, double T0) {
  return C_D(Re(x, gas, d_p, v_p, p0, T0), T_p, T_c(x, gas, T0), M_c(x, gas), gas);
}

// Particle acceleration
double dvdt_p_c(double z, double D, int gas, double d_p, double v_p, double p0, double T0) {
  double x = z_to_x(z, D);
  double dv = v_p - v_g_c(x, gas, T0);
  //printf("x = %g; dv = %g\n", x, dv);
  return C_D_c(x, gas, d_p, v_p, p0, T0) * rho_c(x, gas, p0, T0) * (dv*dv) * M_PI * (d_p*d_p/8.) / m_p(d_p);
}
double ddvdtdz_p_c(double z, double D, int gas, double d_p, double v_p, double p0, double T0, double dz) {
  double ddvdt = dvdt_p_c(z+dz, D, gas, d_p, v_p, p0, T0) - dvdt_p_c(z-dz, D, gas, d_p, v_p, p0, T0);
  return ddvdt/(2.*dz);  
}
// First derivative of particle acceleration
double ddvddt_p_c(double z, double D, int gas, double d_p, double v_p, double p0, double T0) {
  return v_p * ddvdtdz_p_c(z, D, gas, d_p, v_p, p0, T0, 1E-6);
}

PyDoc_STRVAR(test__doc__, "Just a test function.");
static PyObject *test(PyObject *self, PyObject *args, PyObject *kwargs)
{
  double p0, T0, D, d_p, v_p, z_p;
  int gas = 1;

  static char *kwlist[] = {"p0", "T0", "gas", "D", "d_p", "v_p", "z_p", NULL};
  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ddidddd", kwlist, &p0, &T0, &gas, &D, &d_p, &v_p, &z_p)) {
    return NULL;
  }

  double x_t = z_to_x(z_p, D); 

  double M_c_t = M_c(x_t, gas);
  printf("M_c = %g\n", M_c_t);

  double rho0_t = rho0(gas, p0, T0);
  printf("rho0 = %g\n", rho0_t);
  
  double eta_t = eta(T0, gas);
  printf("eta = %g\n", eta_t);
  
  double Re_t = Re(x_t, gas, d_p, v_p, p0, T0);
  printf("Re = %g\n", Re_t);

  double T_c_t = T_c(x_t, gas, T0);
  printf("T_c = %g\n", T_c_t);

  double p_c_t = p_c(x_t, gas, p0);
  printf("p_c = %g\n", p_c_t);

  double rho_c_t = rho_c(x_t, gas, p0, T0);
  printf("rho_c = %g\n", rho_c_t);
 
  double v_g_c_t = v_g_c(x_t, gas, T0);
  printf("v_g_c = %g\n", v_g_c_t);

  double C_D_c_t = C_D_c(x_t, gas, d_p, v_p, p0, T0);
  printf("C_D_c = %g\n", C_D_c_t);

  double C_D_supersonic_t = C_D_supersonic(Re_t, T_p, T_c_t, M_c_t, S(M_c_t, gas));
  printf("C_D_supersonic = %g\n", C_D_supersonic_t);

  double C_D_subsonic_t = C_D_subsonic(Re_t, T_p, T_c_t, M_c_t, S(M_c_t, gas));
  printf("C_D_subsonic = %g\n", C_D_subsonic_t);

  double dvdt_p_c_t = dvdt_p_c(z_p, D, gas, d_p, v_p, p0, T0);
  printf("dvdt_p_c = %g\n", dvdt_p_c_t);

  double ddvdtdz_p_c_t = ddvdtdz_p_c(z_p, D, gas, d_p, v_p, p0, T0, 1e-6);
  printf("ddvdtdz_p_c = %g\n", ddvdtdz_p_c_t);

  double ddvddt_p_c_t = ddvddt_p_c(z_p, D, gas, d_p, v_p, p0, T0);
  printf("ddvddt_p_c = %g\n", ddvddt_p_c_t);
  
  Py_RETURN_NONE;
}


PyDoc_STRVAR(iterate_particle_motion__doc__, "Particle motion in a free jet - expanding gas passing through a nozzle into a low pressure chamber.");
static PyObject *iterate_particle_motion(PyObject *self, PyObject *args, PyObject *kwargs)
{  
  double p0, T0, D, d, v_0, z_max;
  int gas;
  double dz_min = 1E-6;
  double dz_max = 50E-6;
  double z_0 = 0.;
  
  int i, success;
  double dt;
  double tmp_dvdt, tmp_ddvddt;
  double tmp_dz, tmp_dv, tmp_da;
  double g = 0.002;
  
  static char *kwlist[] = {"p0", "T0", "gas", "D", "d", "v_0", "z_max", "dz_min", "dz_max", NULL};
  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ddidddd|dd", kwlist, &p0, &T0, &gas, &D, &d, &v_0, &z_max, &dz_min, &dz_max)) {
    return NULL;
  }

  int N = (int) ceil(z_max / dz_min);
  //printf("N = %d\n", N);

  double * z = (double *) calloc(N, sizeof(double));
  double * v = (double *) calloc(N, sizeof(double));
  double * a = (double *) calloc(N, sizeof(double));

  // Initialise
  z[0] = z_0;
  v[0] = v_0;
  a[0] = ddvddt_p_c(z_0, D, gas, d, v_0, p0, T0);
  z[1] = z[0];
  v[1] = v[0];
  a[1] = a[0];
  
  i = 1;
  int k = 0;
  while ((z[i-1] < z_max)){//&&(k<1)) {
    k++;
    tmp_dvdt = dvdt_p_c(z[i], D, gas, d, v[i], p0, T0);
    tmp_ddvddt = ddvddt_p_c(z[i], D, gas, d, v[i], p0, T0);
    dt = 2 * g * fabs(tmp_dvdt) / fabs(tmp_ddvddt);
    
    success = 0;
    while (!success) {
      tmp_da = dt * tmp_ddvddt;
      tmp_dv = dt * tmp_dvdt + 0.5 * (dt*dt) * tmp_ddvddt;
      tmp_dz = dt * (v[i] + tmp_dv);
      if (tmp_dz < dz_max) {
	success = 1;
      } else {
	dt /= 10.;
      }
    }    
    a[i] = a[i] + tmp_da;
    v[i] = v[i] + tmp_dv;
    z[i] = z[i] + tmp_dz;
    //printf("z = %g\tv = %g\ta = %g\n", z[i], v[i], a[i]);
    if ((z[i]-z[i-1]) >= dz_min) {
      i++;
      //printf("%i/%i\n", i, N);
      a[i] = a[i-1];
      v[i] = v[i-1];
      z[i] = z[i-1];
      if (i >= N) {
	puts("ERROR: Exceeded limits");
	break;
      }
    }
  }

  // Create Python objects for output
  N = i;
  npy_intp dims[1] = {N};
  PyObject *Z = (PyObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  PyObject *V = (PyObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  PyObject *A = (PyObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  double * Zdata = PyArray_DATA(Z);
  double * Vdata = PyArray_DATA(V);
  double * Adata = PyArray_DATA(A);
  for (i=0; i<N; i++) {
    Zdata[i] = z[i];
    Vdata[i] = v[i];
    Adata[i] = a[i];
  }
  free(z);
  free(v);
  free(a);

  PyObject *O = (PyObject *)PyTuple_Pack(3, Z, V, A);
  return O;
}

static PyMethodDef FjMethods[] = {
  {"iterate_particle_motion", (PyCFunction)iterate_particle_motion, METH_VARARGS|METH_KEYWORDS, iterate_particle_motion__doc__},
  {"test", (PyCFunction)test, METH_VARARGS|METH_KEYWORDS, test__doc__},
  {NULL, NULL, 0, NULL}
};

 static struct PyModuleDef fjmodule = {
        PyModuleDef_HEAD_INIT,
        "fj",           
        "Particle motion in a free jet - expanding gas passing through a nozzle into a low pressure chamber..", 
        -1,                   
        FjMethods,       
        NULL,                
        NULL,              
        NULL,                   
        NULL,                   
  };

PyMODINIT_FUNC PyInit_initfj(void)
{
  import_array();
  return PyModule_Create(&fjmodule);
}
