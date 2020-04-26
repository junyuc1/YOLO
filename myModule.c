#include <Python.h>
#include "numpy/arrayobject.h"
#include <math.h>

int not_floatvector(PyArrayObject *vec)  {
	if (vec -> descr -> type_num != NPY_FLOAT)  {
		PyErr_SetString(PyExc_ValueError,
			"In not_float32vector: array must be of type Float.");
		return 1;  }
	return 0;
}

int get_index(int N, int C, int H, int W, int a, int b, int c, int d) {
	return a * C * H * W + b * W * H + c * W + d;
}

static PyObject *Conv_Seq(PyObject *self, PyObject *args) {
	PyArrayObject *inp_p, *weight_p, *bias_p, *result_p;
	float *inp_c, *weight_c, *bias_c, *result_c;

	int N, C, H, W;
	int num_filter, filter_depth, filter_H, filter_W;
	int n, c, h, w, k_d, k_h, k_w;

	int stride;
	int out_size;
	int dims[5];

	if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &inp_p, &PyArray_Type, &weight_p, &PyArray_Type, &bias_p, &stride)) {
		PyErr_SetString(PyExc_ValueError,
			"Parsing Failed.");
		return NULL;
	}
	if (inp_p == NULL || weight_p == NULL || bias_p == NULL || not_floatvector(inp_p) || not_floatvector(weight_p) || not_floatvector(bias_p)) {
		PyErr_SetString(PyExc_ValueError,
			"Input Invalid.");
		return NULL;
	}

	N = inp_p -> dimensions[0];
	C = inp_p -> dimensions[1];
	H = inp_p -> dimensions[2];
	W = inp_p -> dimensions[3];
	num_filter = weight_p -> dimensions[0];
	filter_depth = weight_p -> dimensions[1];
	filter_H = weight_p -> dimensions[2];
	filter_W = weight_p -> dimensions[3];

	out_size = floor((W - filter_W) / stride) + 1;

	dims[0] = N;
	dims[1] = num_filter;
	dims[2] = out_size;
	dims[3] = out_size;
	result_p = (PyArrayObject *)(PyArray_FromDims(4, dims, NPY_FLOAT));

	inp_c = (float *)(inp_p -> data);
	weight_c = (float *)(weight_p -> data);
	bias_c = (float *)(bias_p -> data);
	result_c = (float *)(result_p -> data);

	for (n = 0; n < N; n ++) {
		for (k_d = 0; k_d < num_filter; k_d ++) {
			for (h = 0; h < out_size; h ++) {
				for (w = 0; w < out_size; w ++) {
					float cur_val = 0.0;
					int kernel_x = w * stride;
					int kernel_y = h * stride;
					for (c = 0; c < filter_depth; c ++) { 
						for (k_h = 0; k_h < filter_H; k_h ++) {
							for (k_w = 0; k_w < filter_W; k_w ++) {
								float target_weight = weight_c[get_index(num_filter, filter_depth, filter_H, filter_W, k_d, c, k_h, k_w)];
								float target_inp = inp_c[get_index(N, C, H, W, n, c, kernel_y + k_h, kernel_x + k_w)];
								cur_val += target_inp * target_weight;
							}
						}
					}
					cur_val += bias_c[k_d];
					result_c[get_index(N, num_filter, out_size, out_size, n, k_d, h, w)] = cur_val;
				}
			}
		}
	}
	return PyArray_Return(result_p);
}

static PyMethodDef myMethods[] = {
	{"Conv_Seq", Conv_Seq, METH_VARARGS},
	{NULL, NULL}
};

static struct PyModuleDef myModule = {
	PyModuleDef_HEAD_INIT,
	"myModule",
	"YOLO",
	-1,
	myMethods
};

PyMODINIT_FUNC PyInit_myModule(void) {
	import_array();
	return PyModule_Create(&myModule);
}
 
