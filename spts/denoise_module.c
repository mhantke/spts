#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>

PyDoc_STRVAR(calc_hists__doc__, "calc_hists(images, vmin, vmax, window_size, dx=1)\nRun this function on a stack of background images. vmin and vmax are the expected lowest and largest value that are within the noise. The window_size should not exceed the smallest feature that shall be detected from the denoised images.\n");
static PyObject *calc_hists(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject *images_obj;
  int vmin;
  int vmax;
  int window_size;
  int v;
  int i,j,k,l;
  int i_hist;
  int x_corner,y_corner;
  int i_x, i_y;
  int N_x, N_y, N_pix, N_images, hist_len;
  int dx = 1;
  
  static char *kwlist[] = {"images", "vmin", "vmax", "window_size", "dx", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiii|i", kwlist, &images_obj, &vmin, &vmax, &window_size, &dx)) {
    return NULL;
  }
  
  PyObject * images_array = PyArray_FROM_OTF(images_obj, NPY_INT, NPY_IN_ARRAY);
  if (images_array == NULL) {
    Py_XDECREF(images_array);
    return NULL;
  }
  int * images = PyArray_DATA(images_array);

  int ndim = PyArray_NDIM(images_array);
  if (ndim != 3) {
    PyErr_SetString(PyExc_ValueError, "Input array must be 3-dimensional\n");
    Py_XDECREF(images_array);
    return NULL;
  }

  N_x      = (int) PyArray_DIM(images_array, 2);
  N_y      = (int) PyArray_DIM(images_array, 1);
  N_pix    = N_x * N_y;
  N_images = (int) PyArray_DIM(images_array, 0);
  hist_len = (vmax - vmin + dx) / dx;
 
  // Build pixel histograms
  int * pixel_hists = (int *) calloc(N_pix*hist_len, sizeof(int));
  for (i = 0; i < N_images; i++) {
    for (j = 0; j < N_pix; j++) {
      v = images[i*N_pix+j];
      if ((v >= vmin) && (v <= vmax)) {
	l = (v-vmin) / dx;
	pixel_hists[j*hist_len+l] += 1;
      } 
    }
  }

  // Combine to window histograms
  int out_dim[] = {N_y, N_x, hist_len};
  PyObject *window_hists_array = (PyObject *)PyArray_FromDims(3, out_dim, NPY_DOUBLE);
  double * window_hists = PyArray_DATA(window_hists_array);
  for (j = 0; j < N_pix; j++) {
    x_corner = j % N_x - window_size/2;
    y_corner = j / N_x - window_size/2;
    if ((x_corner >= 0) && ((x_corner+window_size) <= N_x) && \
	(y_corner >= 0) && ((y_corner+window_size) <= N_y)) {
      for (i_x = 0; i_x < window_size; i_x++) {
	for (i_y = 0; i_y < window_size; i_y++) {
	  i_hist = (y_corner + i_y) * N_x + (x_corner + i_x);
	  for (k = 0; k < hist_len; k++) {
	    window_hists[j*hist_len+k] += pixel_hists[i_hist*hist_len+k];
	  }
	}
      }
    }
  }

  // Normalise to one frame
  for (j = 0; j < N_pix; j++) {
    for (k = 0; k < hist_len; k++) {
      window_hists[j*hist_len+k] /= N_images;
    }
  }
  
  free(pixel_hists);
  
  return window_hists_array;
}

PyDoc_STRVAR(denoise__doc__, "denoise(image, hists_bg, vmin, vmax, window_size, vmin_full=-50, vmax_full=255, dx=1)\n\n");
static PyObject *denoise(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject *image_obj;
  PyObject *hists_bg_obj;
  int vmin;
  int vmax;
  int window_size;
  int v;
  int i,j,k,l;
  int x_corner,y_corner;
  int i_x, i_y;
  int n_bg, n_img;
  // Set default values for not provided optional keyword arguments
  int vmin_full = -50;
  int vmax_full = 255;
  int dx = 1;
  
  static char *kwlist[] = {"image", "hists_bg", "vmin", "vmax", "window_size", "vmin_full", "vmax_full", "dx", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOiii|iii", kwlist, &image_obj, &hists_bg_obj, &vmin, &vmax, &window_size, &vmin_full, &vmax_full, &dx)) {
    return NULL;
  }
  
  PyObject * image_array = PyArray_FROM_OTF(image_obj, NPY_INT, NPY_IN_ARRAY);
  if (image_array == NULL) {
    Py_XDECREF(image_array);
    return NULL;
  }
  int * image = PyArray_DATA(image_array);

  int ndim_image = PyArray_NDIM(image_array);
  if (ndim_image != 2) {
    PyErr_SetString(PyExc_ValueError, "Image input array must be 2-dimensional\n");
    Py_XDECREF(image_array);
    return NULL;
  }

  PyObject * hists_bg_array = PyArray_FROM_OTF(hists_bg_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  if (hists_bg_array == NULL) {
    Py_XDECREF(hists_bg_array);
    return NULL;
  }
  double * hists_bg = PyArray_DATA(hists_bg_array);

  int ndim_hists_bg = PyArray_NDIM(hists_bg_array);
  if (ndim_hists_bg != 3) {
    PyErr_SetString(PyExc_ValueError, "Hists_Bg input array must be 3-dimensional\n");
    Py_XDECREF(hists_bg_array);
    Py_XDECREF(image_array);
    return NULL;
  }

  int N_x      = (int) PyArray_DIM(hists_bg_array, 1);
  int N_y      = (int) PyArray_DIM(hists_bg_array, 0);
  int N_pix    = N_x * N_y;
  int full_hist_len = (vmax_full - vmin_full + dx) / dx;
  int hist_len = (vmax - vmin + dx) / dx;

  //printf("full_hist_len = %d\n", full_hist_len);
  //printf("hist_len = %d\n", hist_len);
  
  // Build window pixel histograms from image
  int * hists_image = (int *) calloc(N_pix*full_hist_len, sizeof(int));
  for (j = 0; j < N_pix; j++) {
    x_corner = j % N_x - window_size/2;
    y_corner = j / N_x - window_size/2;
    if ((x_corner >= 0) && ((x_corner+window_size) <= N_x) && \
	(y_corner >= 0) && ((y_corner+window_size) <= N_y)) {
      for (i_x = 0; i_x < window_size; i_x++) {
	for (i_y = 0; i_y < window_size; i_y++) {
	  i = (y_corner + i_y) * N_x + (x_corner + i_x);
	  v = image[i];
	  k = (v - vmin_full) / dx;
	  if ((v >= vmin_full) && (v <= vmax_full)) {
	    //printf("j = %d (%d) ; v = %d ; k = %d (%d)\n", j, N_pix, v, k, full_hist_len);
	    //printf("index = %d (%d)\n", j*full_hist_len+k, N_pix*full_hist_len);
	    hists_image[j*full_hist_len+k] += 1;
	  } else if (v > vmax_full) {
	    // If above range add to last bin
	    k = full_hist_len - 1;
	    hists_image[j*full_hist_len+k] += 1;
	  }
	}
      }
    }
  }

  //puts("Guard 1");
  
  // Compare and score
  int out_dim[] = {N_y, N_x};
  PyObject * scores_array = (PyObject *)PyArray_FromDims(2, out_dim, NPY_DOUBLE);
  double * scores = PyArray_DATA(scores_array);
  for (j = 0; j < N_pix; j++) {
    scores[j] = 0.;
    for (k = 0; k < full_hist_len; k++) {
      v = vmin_full + k * dx;
      if ((v >= vmin) && (v <= vmax)) {
	l = (v-vmin)/dx;
	n_bg = hists_bg[j*hist_len+l];
      } else {
	n_bg = 0;
      }
      n_img = hists_image[j*full_hist_len+k];
      // Weighted subtraction of histograms
      scores[j] += v * (n_img - n_bg);
    }
  }
  
  free(hists_image);

  //puts("Guard 2");
  
  return scores_array;
}


static PyMethodDef DenoiseMethods[] = {
  {"calc_hists", (PyCFunction)calc_hists, METH_VARARGS|METH_KEYWORDS, calc_hists__doc__},
  {"denoise", (PyCFunction)denoise, METH_VARARGS|METH_KEYWORDS, denoise__doc__},
  {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC initdenoise(void)
{
  import_array();
  PyObject *m = Py_InitModule3("denoise", DenoiseMethods, "Denoise image stack.");
  if (m == NULL)
    return;
}
