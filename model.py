from __future__ import division
import numpy as np
from timeit import default_timer as timer
import multiprocessing
from multiprocessing import Process, Lock, Pool
import threading
import concurrent.futures
import myModule
import myModule_CUDA

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
	N, C, H, W = x_shape
	out_height = (H + 2 * padding - field_height) // stride + 1
	out_width = (W + 2 * padding - field_width) // stride + 1

	i0 = np.repeat(np.arange(field_height), field_width)
	i0 = np.tile(i0, C)
	i1 = stride * np.repeat(np.arange(out_height), out_width)
	j0 = np.tile(np.arange(field_width), field_height * C)
	j1 = stride * np.tile(np.arange(out_width), out_height)
	i = i0.reshape(-1, 1) + i1.reshape(1, -1)
	j = j0.reshape(-1, 1) + j1.reshape(1, -1)

	k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

	return (k, i, j)

def im2col(x, field_height, field_width, padding=1, stride=1):
	p = padding
	x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

	k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

	cols = x_padded[:, k, i, j]
	C = x.shape[1]
	cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
	return cols

class Convolution:
	def __init__(self, depth, filters, size, stride, pad):
		self.depth = depth
		self.filters = filters
		self.size = size
		self.stride = stride
		self.pad = pad
		self.weights = np.zeros((filters, depth, size, size))
		self.bias = np.zeros((filters, 1), dtype=np.float32)

	def load_weights(self, weights):
		weights = np.reshape(weights, (self.filters, self.depth, self.size, self.size))
		self.weights = weights

	def load_bias(self, bias):
		bias = np.reshape(bias, (self.filters, 1))
		self.bias = bias

	def forward_seq(self, input):
		inp = np.pad(input, [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], 'constant', constant_values=0)
		batch_size, depth, width, height = inp.shape
		out_width = (width - self.size) // self.stride + 1
		out_height = (height - self.size) // self.stride + 1
		result = np.zeros((batch_size, self.filters, out_width, out_height))
		for b in range(batch_size):
			for d in range(self.filters):
				for h in range(out_height):
					for w in range(out_width):
						cur_val = 0.0
						kernel_x = w * self.stride
						kernel_y = h * self.stride
						for kd in range(self.depth):
							for r in range(self.size):
								for c in range(self.size):
									cur_val += self.weights[d][kd][r][c] * inp[b][kd][kernel_y + r][kernel_x + c]
									result[b][d][h][w] = cur_val
		return result
	
	def forward(self, inputs):
		weight_flat = self.weights.reshape(self.filters, -1)
		input_flat = im2col(inputs, self.size, self.size, self.pad, self.stride)
		flat_feature = np.matmul(weight_flat, input_flat) + self.bias
		out_width = (inputs.shape[3] + 2 * self.pad - self.size) // self.stride + 1
		out_height = (inputs.shape[2] + 2 * self.pad - self.size) // self.stride + 1
		flat_feature_trans = flat_feature.reshape(self.filters, out_height, out_width, inputs.shape[0])
		feature = flat_feature_trans.transpose(3, 0, 1, 2)
		return feature

	def forward_c_seq(self, inputs):
		inp = np.pad(inputs, [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], 'constant', constant_values=0.0)
		return myModule.Conv_Seq(inp, self.weights, self.bias, self.stride)

	def forward_c_par_omp(self, inputs):
		inp = np.pad(inputs, [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], 'constant', constant_values=0.0)
		return myModule.Conv_Par_OMP(inp, self.weights, self.bias, self.stride)

	def forward_c_par_cuda(self, inputs):
		inp = np.pad(inputs, [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], 'constant', constant_values=0.0)
		bias = np.reshape(self.bias, self.filters)
		out_size = (inp.shape[3] - self.weights.shape[3]) // self.stride + 1
		dims = np.array(list(inp.shape) + list(self.weights.shape) + [out_size])
		dims = dims.astype('float32')
		return myModule_CUDA.Conv_Par_CUDA(inp, self.weights, bias, dims, self.stride)

	def forward_c_par_matmul_cuda(self, inputs):
		weight_flat = self.weights.reshape(self.filters, -1)
		input_flat = im2col(inputs, self.size, self.size, self.pad, self.stride)
		flat_feature = myModule_CUDA.Matmul_Par_CUDA(weight_flat, input_flat, weight_flat.shape[0], weight_flat.shape[1], input_flat.shape[0], input_flat.shape[1]) + self.bias 
		out_width = (inputs.shape[3] + 2 * self.pad - self.size) // self.stride + 1
		out_height = (inputs.shape[2] + 2 * self.pad - self.size) // self.stride + 1
		flat_feature_trans = flat_feature.reshape(self.filters, out_height, out_width, inputs.shape[0])
		feature = flat_feature_trans.transpose(3, 0, 1, 2)
		return feature

class Batch_Normalize:
	def __init__(self, filters):
		self.filters = filters
		self.eps = 1e-05
		self.momentum = 0.1

	def load_weights(self, weights):
		self.weights = np.reshape(weights, (1, self.filters, 1, 1))

	def load_bias(self, bias):
		self.bias = np.reshape(bias, (1, self.filters, 1, 1))

	def load_running_mean(self, running_mean):
		self.running_mean = np.reshape(running_mean, (1, self.filters, 1, 1))

	def load_running_var(self, running_var):
		self.running_var = np.reshape(running_var, (1, self.filters, 1, 1)) 

	def forward(self, inputs):
		mean = np.reshape(np.mean(inputs, axis=(0, 2, 3)), (1, self.filters, 1, 1))
		variance = np.reshape(np.mean((inputs - mean.reshape((1, self.filters, 1, 1))) ** 2, axis=(0, 2, 3)), (1, self.filters, 1, 1))
		inputs_est = (inputs - mean) / np.sqrt(variance + self.eps)
		res = self.weights * inputs_est + self.bias
		return res
		
class LRelu:
	def __init__(self, negative_slope):
		self.negative_slope = negative_slope

	def forward(self, input):
		input[input < 0] = input[input < 0] * self.negative_slope
		return input 

class Conv_Block:
	def __init__(self):
		self.models = []
		self.num_models = 0

	def add_model(self, model):
		self.models.append(model)
		self.num_models += 1

	def forward(self, input):
		output = self.models[0].forward_c_par_cuda(input)
		for i in range(1, self.num_models):
			output = self.models[i].forward(output)
		return output

class Upsample:
	def __init__(self, repeat):
		self.repeat = repeat

	def forward(self, input):
		return input.repeat(self.repeat, axis=2).repeat(self.repeat, axis=3)

class Route:
	def __init__(self, layers):
		self.layers = layers

	def forward(self, layer_outputs):
		if len(self.layers) == 1:
			return layer_outputs[self.layers[0]]
		else:
			return np.concatenate((layer_outputs[self.layers[0]], layer_outputs[self.layers[1]]), axis=1)

class Shortcut:
	def __init__(self, pos1, pos2):
		self.pos1 = pos1
		self.pos2 = pos2

	def forward(self, layer_outputs):
		return layer_outputs[self.pos1] + layer_outputs[self.pos2]

class Yolo:
	def __init__(self, anchors, num_class):
		self.anchors = anchors
		self.num_class = num_class

	def forward(self, image_size, output):
		image_size = int(image_size)
		batch, total_attr, num_grid, num_grid = output.shape
		grid_len = image_size // num_grid
		num_grid = image_size // grid_len
		num_attr = 5 + self.num_class
		num_anchors = len(self.anchors)

		pred = np.reshape(output, (batch, total_attr, num_grid * num_grid))
		pred = np.transpose(pred, (0, 2, 1))
		pred = np.reshape(pred, (batch, num_grid * num_grid * num_anchors, num_attr))
		anchors = [[a[0] / grid_len, a[1] / grid_len] for a in self.anchors]

		pred[:,:,0] = 1.0 / (1.0 + np.exp(- pred[:,:,0]))
		pred[:,:,1] = 1.0 / (1.0 + np.exp(- pred[:,:,1]))
		pred[:,:,4] = 1.0 / (1.0 + np.exp(- pred[:,:,4]))
		
		grid = np.arange(num_grid)
		a, b = np.meshgrid(grid, grid)

		x_offset = np.reshape(a, (-1, 1))
		y_offset = np.reshape(b, (-1, 1))

		x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
		x_y_offset = np.repeat(x_y_offset, num_anchors, axis=0)
		x_y_offset = np.array([np.reshape(x_y_offset, (-1, 2))])

		pred[:,:,:2] += x_y_offset
		
		anchors = np.array(anchors)

		anchors = np.tile(anchors, (num_grid * num_grid, 1))
		anchors = np.array([anchors])
		
		pred[:,:,2:4] = np.multiply(np.exp(pred[:,:,2:4]), anchors)
		pred[:,:,5: 5 + self.num_class] = 1.0 / (1.0 + np.exp(- pred[:,:,5: 5 + self.num_class]))
		pred[:,:,:4] *= grid_len

		return pred


