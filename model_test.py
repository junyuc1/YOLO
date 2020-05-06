from __future__ import division
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import * 
from darknet_impl import *
import time
import multiprocessing
from multiprocessing import Process, Lock, Pool
import threading
import concurrent.futures
from model import *
import myModule
import myModule_CUDA

def check_conv(torch_out, my_out):
	torch_out = torch_out.data.numpy()
	rtol = 1e-04
	atol = 1e-04
	return np.allclose(torch_out, my_out, rtol, atol)

my_inp = get_input("image/eagle.jpg")
torch_inp = torch.from_numpy(my_inp)

fp = open("yolov3.weights", "rb")
weights = np.fromfile(fp, dtype = np.float32)

def test_conv_1():
	my_conv = Convolution(3, 32, 3, 1, 1)
	torch_conv = nn.Conv2d(3, 32, 3, 1, 1, bias = False)

	my_weights = weights[128:992]
	torch_weights = torch.from_numpy(weights[128:992])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)

	torch_result = torch_conv(torch_inp)
	my_result = my_conv.forward_c_par_matmul_cuda(my_inp)

	if check_conv(torch_result, my_result):
		print("test conv 1 passed")

def test_conv_1_c_seq():
	my_conv = Convolution(3, 32, 3, 1, 1)
	torch_conv = nn.Conv2d(3, 32, 3, 1, 1, bias = False)

	my_weights = weights[128:992]
	torch_weights = torch.from_numpy(weights[128:992])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)

	torch_result = torch_conv(torch_inp)
	my_result = my_conv.forward_c_seq(my_inp)

	if check_conv(torch_result, my_result):
		print("test conv c seq 1 passed")

def test_conv_c_par_cuda():
	my_inp = cv2.imread("image/dog.jpg")
	my_inp = np.array(prep_image(my_inp, 416))[:,:2,:3,:3]
	torch_inp = torch.from_numpy(my_inp)

	my_conv = Convolution(2, 2, 2, 1, 1)
	torch_conv = nn.Conv2d(2, 2, 2, 1, 1, bias = False)

	my_weights = weights[128:144]
	torch_weights = torch.from_numpy(weights[128:144])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)
	

	torch_result = torch_conv(torch_inp)
	
	my_result = my_conv.forward_c_par_matmul_cuda(my_inp)
	
	print(my_result)
	if check_conv(torch_result, my_result):
		print("test conv c par cuda passed")

def test_conv_c_par_omp():
	my_inp = []
	for i in range(4):
		im1 = cv2.imread("image/dog.jpg")
		im1 = np.array(prep_image(im1, 416))[0]
		im2 = cv2.imread("image/eagle.jpg")
		im2 = np.array(prep_image(im2, 416))[0]
		im3 = cv2.imread("image/giraffe.jpg")
		im3 = np.array(prep_image(im3, 416))[0]
		im4 = cv2.imread("image/herd_of_horses.jpg")
		im4 = np.array(prep_image(im4, 416))[0]
		im5 = cv2.imread("image/img1.jpg")
		im5 = np.array(prep_image(im5, 416))[0]
		im6 = cv2.imread("image/img2.jpg")
		im6 = np.array(prep_image(im6, 416))[0]
		im7 = cv2.imread("image/img3.jpg")
		im7 = np.array(prep_image(im7, 416))[0]
		im8 = cv2.imread("image/img4.jpg")
		im8 = np.array(prep_image(im8, 416))[0]
		my_inp.append(im1)
		my_inp.append(im2)
		my_inp.append(im3)
		my_inp.append(im4)
		my_inp.append(im5)
		my_inp.append(im6)
		my_inp.append(im7)
		my_inp.append(im8)
	my_inp = np.array(my_inp)
	my_inp = np.ones((8, 32, 416, 416), dtype="float32")
	torch_inp = torch.from_numpy(my_inp)

	my_conv = Convolution(32, 64, 3, 2, 1)
	torch_conv = nn.Conv2d(32, 64, 3, 2, 1, bias = True)

	my_weights = weights[128:18560]
	torch_weights = torch.from_numpy(weights[128:18560])

	my_bias = weights[1000:1064]
	torch_bias = torch.from_numpy(weights[1000:1064])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)

	torch_conv.bias.data.copy_(torch_bias.view_as(torch_conv.bias.data))
	my_conv.load_bias(my_bias)


	start = time.time()
	torch_result = torch_conv(torch_inp)
	
	end = time.time()
	print("torch time " + str(end - start))

	start = time.time()
	my_result = my_conv.forward_c_par_cuda(my_inp)
	
	end = time.time()
	print("my time " + str(end - start))

	if check_conv(torch_result, my_result):
		print("passed")


def test_conv_2():
	my_conv = Convolution(3, 32, 3, 1, 5)
	torch_conv = nn.Conv2d(3, 32, 3, 1, 5, bias = False)

	my_weights = weights[128:992]
	torch_weights = torch.from_numpy(weights[128:992])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)

	torch_result = torch_conv(torch_inp)
	my_result = my_conv.forward(my_inp)

	if check_conv(torch_result, my_result):
		print("test conv 2 passed")

def test_conv_2_c_seq():
	my_conv = Convolution(3, 32, 3, 1, 5)
	torch_conv = nn.Conv2d(3, 32, 3, 1, 5, bias = False)

	my_weights = weights[128:992]
	torch_weights = torch.from_numpy(weights[128:992])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)

	torch_result = torch_conv(torch_inp)
	my_result = my_conv.forward_c_seq(my_inp)

	if check_conv(torch_result, my_result):
		print("test conv c seq 2 passed")

def test_conv_2_c_par_cuda():
	my_conv = Convolution(3, 32, 3, 1, 5)
	torch_conv = nn.Conv2d(3, 32, 3, 1, 5, bias = False)

	my_weights = weights[128:992]
	torch_weights = torch.from_numpy(weights[128:992])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)

	torch_result = torch_conv(torch_inp)
	my_result = my_conv.forward_c_par_cuda(my_inp)

	if check_conv(torch_result, my_result):
		print("test conv c seq 2 passed")

def test_conv_3():
	my_conv = Convolution(3, 32, 3, 1, 5)
	torch_conv = nn.Conv2d(3, 32, 3, 1, 5, bias = True)

	my_weights = weights[128:992]
	torch_weights = torch.from_numpy(weights[128:992])

	my_bias = weights[128:160]
	torch_bias = torch.from_numpy(weights[128:160])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)

	torch_conv.bias.data.copy_(torch_bias.view_as(torch_conv.bias.data))
	my_conv.load_bias(my_bias)

	torch_result = torch_conv(torch_inp)
	my_result = my_conv.forward(my_inp)

	if check_conv(torch_result, my_result):
		print("test conv 3 passed")

def test_conv_3_c_seq():
	my_conv = Convolution(3, 32, 3, 1, 5)
	torch_conv = nn.Conv2d(3, 32, 3, 1, 5, bias = True)

	my_weights = weights[128:992]
	torch_weights = torch.from_numpy(weights[128:992])

	my_bias = weights[128:160]
	torch_bias = torch.from_numpy(weights[128:160])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)

	torch_conv.bias.data.copy_(torch_bias.view_as(torch_conv.bias.data))
	my_conv.load_bias(my_bias)

	torch_result = torch_conv(torch_inp)
	my_result = my_conv.forward_c_seq(my_inp)

	if check_conv(torch_result, my_result):
		print("test conv c seq 3 passed")

def test_conv_3_c_par_cuda():
	my_conv = Convolution(3, 32, 3, 1, 5)
	torch_conv = nn.Conv2d(3, 32, 3, 1, 5, bias = True)

	my_weights = weights[128:992]
	torch_weights = torch.from_numpy(weights[128:992])

	my_bias = weights[128:160]
	torch_bias = torch.from_numpy(weights[128:160])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)

	torch_conv.bias.data.copy_(torch_bias.view_as(torch_conv.bias.data))
	my_conv.load_bias(my_bias)

	torch_result = torch_conv(torch_inp)
	my_result = my_conv.forward_c_par_cuda(my_inp)

	if check_conv(torch_result, my_result):
		print("test conv c seq 3 passed")

def test_conv_4():
	my_conv = Convolution(3, 32, 3, 1, 0)
	torch_conv = nn.Conv2d(3, 32, 3, 1, 0, bias = True)

	my_weights = weights[128:992]
	torch_weights = torch.from_numpy(weights[128:992])

	my_bias = weights[128:160]
	torch_bias = torch.from_numpy(weights[128:160])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)

	torch_conv.bias.data.copy_(torch_bias.view_as(torch_conv.bias.data))
	my_conv.load_bias(my_bias)

	torch_result = torch_conv(torch_inp)
	my_result = my_conv.forward(my_inp)

	if check_conv(torch_result, my_result):
		print("test conv 4 passed")

def test_conv_4_c_seq():
	my_conv = Convolution(3, 32, 3, 1, 0)
	torch_conv = nn.Conv2d(3, 32, 3, 1, 0, bias = True)

	my_weights = weights[128:992]
	torch_weights = torch.from_numpy(weights[128:992])

	my_bias = weights[128:160]
	torch_bias = torch.from_numpy(weights[128:160])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)

	torch_conv.bias.data.copy_(torch_bias.view_as(torch_conv.bias.data))
	my_conv.load_bias(my_bias)

	torch_result = torch_conv(torch_inp)
	my_result = my_conv.forward_c_seq(my_inp)

	if check_conv(torch_result, my_result):
		print("test conv c seq 4 passed")

def test_conv_4_c_par_cuda():
	my_conv = Convolution(3, 32, 3, 1, 0)
	torch_conv = nn.Conv2d(3, 32, 3, 1, 0, bias = True)

	my_weights = weights[128:992]
	torch_weights = torch.from_numpy(weights[128:992])

	my_bias = weights[128:160]
	torch_bias = torch.from_numpy(weights[128:160])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)

	torch_conv.bias.data.copy_(torch_bias.view_as(torch_conv.bias.data))
	my_conv.load_bias(my_bias)

	torch_result = torch_conv(torch_inp)
	my_result = my_conv.forward_c_par_cuda(my_inp)

	if check_conv(torch_result, my_result):
		print("test conv c seq 4 passed")

def test_bn_1():
	my_bn = Batch_Normalize(3)
	torch_bn = nn.BatchNorm2d(3)

	my_bias = weights[128:131]
	torch_bias = torch.from_numpy(weights[128:131])

	my_weights = weights[131:134]
	torch_weights = torch.from_numpy(weights[131:134])

	my_running_mean = weights[134:137]
	torch_running_mean = torch.from_numpy(weights[134:137])

	my_running_var = weights[137:140]
	torch_running_var = torch.from_numpy(weights[137:140])

	torch_bn.bias.data.copy_(torch_bias.view_as(torch_bn.bias.data))
	torch_bn.weight.data.copy_(torch_weights.view_as(torch_bn.weight.data))
	torch_bn.running_mean.copy_(torch_running_mean.view_as(torch_bn.running_mean.data))
	torch_bn.running_var.copy_(torch_running_var.view_as(torch_bn.running_var.data))

	my_bn.load_bias(my_bias)
	my_bn.load_weights(my_weights)
	my_bn.load_running_mean(my_running_mean)
	my_bn.load_running_var(my_running_var)

	torch_result = torch_bn(torch_inp)
	my_result = my_bn.forward(my_inp)

	if check_conv(torch_result, my_result):
		print("test bn 1 passed")

def test_bn_2():
	my_bn = Batch_Normalize(3)
	torch_bn = nn.BatchNorm2d(3)

	my_bias = weights[128:131]
	torch_bias = torch.from_numpy(weights[128:131])

	my_weights = weights[131:134]
	torch_weights = torch.from_numpy(weights[131:134])

	my_running_mean = weights[134:137]
	torch_running_mean = torch.from_numpy(weights[134:137])

	my_running_var = weights[137:140]
	torch_running_var = torch.from_numpy(weights[137:140])

	torch_bn.bias.data.copy_(torch_bias.view_as(torch_bn.bias.data))
	torch_bn.weight.data.copy_(torch_weights.view_as(torch_bn.weight.data))

	my_bn.load_bias(my_bias)
	my_bn.load_weights(my_weights)
	my_bn.load_running_mean(my_running_mean)
	my_bn.load_running_var(my_running_var)

	torch_result = torch_bn(torch_inp)
	my_result = my_bn.forward(my_inp)

	if check_conv(torch_result, my_result):
		print("test bn 2 passed")

def test_bn_3():
	my_bn = Batch_Normalize(3)
	torch_bn = nn.BatchNorm2d(3)

	my_bias = weights[128:131]
	torch_bias = torch.from_numpy(weights[128:131])

	my_weights = weights[131:134]
	torch_weights = torch.from_numpy(weights[131:134])

	my_running_mean = weights[134:137]
	torch_running_mean = torch.from_numpy(weights[134:137])

	my_running_var = weights[137:140]
	torch_running_var = torch.from_numpy(weights[137:140])

	torch_bn.bias.data.copy_(torch_bias.view_as(torch_bn.bias.data))
	torch_bn.weight.data.copy_(torch_weights.view_as(torch_bn.weight.data))
	torch_bn.running_mean.copy_(torch_running_mean.view_as(torch_bn.running_mean.data))
	torch_bn.running_var.copy_(torch_running_var.view_as(torch_bn.running_var.data))

	my_bn.load_bias(my_bias)
	my_bn.load_weights(my_weights)
	my_bn.load_running_mean(my_running_mean)
	my_bn.load_running_var(my_running_var)

	torch_result = torch_bn(torch_bn(torch_bn(torch_inp)))
	my_result = my_bn.forward(my_bn.forward(my_bn.forward(my_inp)))

	torch_result2 = torch_bn(torch_inp)
	my_result2 = my_bn.forward(my_inp)
	if check_conv(torch_result, my_result) and check_conv(torch_result2, my_result2):
		print("test bn 3 passed")

def test_lrelu_1():
	my_inp = get_input()
	torch_inp = get_test_input()
	my_in = my_inp
	my_in[:, :, 0:300, :49] = - my_in[:, :, 0:300, :49]
	torch_in = torch_inp
	torch_in[:, :, 0:300, :49] = - torch_in[:, :, 0:300, :49]

	torch_lrelu = nn.LeakyReLU(0.1, inplace = True)
	my_lrelu = LRelu(0.1)

	torch_result = torch_lrelu(torch_in)
	my_result = my_lrelu.forward(my_in)

	if check_conv(torch_result, my_result):
		print("test lrelu 1 passed")

def test_lrelu_2():
	my_inp = get_input()
	torch_inp = get_test_input()
	my_in = my_inp
	my_in[:, :, 0:300, :49] = - my_in[:, :, 0:300, :49]
	torch_in = torch_inp
	torch_in[:, :, 0:300, :49] = - torch_in[:, :, 0:300, :49]

	torch_lrelu = nn.LeakyReLU(0.1, inplace = True)
	my_lrelu = LRelu(0.1)

	torch_result = torch_lrelu(torch_in)
	my_result = my_lrelu.forward(my_in)

	if check_conv(torch_in, my_in):
		print("test lrelu 2 passed")

def test_conv_blk_1():
	my_inp = get_input()
	torch_inp = get_test_input()

	my_blk = Conv_Block()
	torch_blk = nn.Sequential()

	my_conv = Convolution(3, 32, 3, 1, 1)
	torch_conv = nn.Conv2d(3, 32, 3, 1, 1, bias = False)

	my_weights = weights[128:992]
	torch_weights = torch.from_numpy(weights[128:992])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)

	my_blk.add_model(my_conv)
	torch_blk.add_module("conv_{0}".format(0), torch_conv)

	my_result = my_blk.forward(my_inp)
	torch_result = torch_blk(torch_inp)

	if check_conv(torch_result, my_result):
		print("test conv blk 1 passed")

def test_conv_blk_2():
	my_inp = get_input()
	torch_inp = get_test_input()

	my_blk = Conv_Block()
	torch_blk = nn.Sequential()

	my_conv = Convolution(3, 32, 3, 1, 1)
	torch_conv = nn.Conv2d(3, 32, 3, 1, 1, bias = False)

	my_weights = weights[128:992]
	torch_weights = torch.from_numpy(weights[128:992])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)

	my_blk.add_model(my_conv)
	torch_blk.add_module("conv_{0}".format(0), torch_conv)

	my_bn = Batch_Normalize(32)
	torch_bn = nn.BatchNorm2d(32)

	my_bias = weights[128:160]
	torch_bias = torch.from_numpy(weights[128:160])

	my_weights = weights[160:192]
	torch_weights = torch.from_numpy(weights[160:192])

	my_running_mean = weights[192:224]
	torch_running_mean = torch.from_numpy(weights[192:224])

	my_running_var = weights[224:256]
	torch_running_var = torch.from_numpy(weights[224:256])

	torch_bn.bias.data.copy_(torch_bias.view_as(torch_bn.bias.data))
	torch_bn.weight.data.copy_(torch_weights.view_as(torch_bn.weight.data))
	torch_bn.running_mean.copy_(torch_running_mean.view_as(torch_bn.running_mean.data))
	torch_bn.running_var.copy_(torch_running_var.view_as(torch_bn.running_var.data))

	my_bn.load_bias(my_bias)
	my_bn.load_weights(my_weights)
	my_bn.load_running_mean(my_running_mean)
	my_bn.load_running_var(my_running_var)

	my_blk.add_model(my_bn)
	torch_blk.add_module("batch_norm_{0}".format(0), torch_bn)

	my_result = my_blk.forward(my_inp)
	torch_result = torch_blk(torch_inp)

	if check_conv(torch_result, my_result):
		print("test conv blk 2 passed")

def test_conv_blk_3():
	my_inp = get_input()
	torch_inp = get_test_input()

	my_blk = Conv_Block()
	torch_blk = nn.Sequential()

	my_conv = Convolution(3, 32, 3, 1, 1)
	torch_conv = nn.Conv2d(3, 32, 3, 1, 1, bias = False)

	my_weights = weights[128:992]
	torch_weights = torch.from_numpy(weights[128:992])

	torch_conv.weight.data.copy_(torch_weights.view_as(torch_conv.weight.data))
	my_conv.load_weights(my_weights)

	my_blk.add_model(my_conv)
	torch_blk.add_module("conv_{0}".format(0), torch_conv)

	my_bn = Batch_Normalize(32)
	torch_bn = nn.BatchNorm2d(32)

	my_bias = weights[128:160]
	torch_bias = torch.from_numpy(weights[128:160])

	my_weights = weights[160:192]
	torch_weights = torch.from_numpy(weights[160:192])

	my_running_mean = weights[192:224]
	torch_running_mean = torch.from_numpy(weights[192:224])

	my_running_var = weights[224:256]
	torch_running_var = torch.from_numpy(weights[224:256])

	torch_bn.bias.data.copy_(torch_bias.view_as(torch_bn.bias.data))
	torch_bn.weight.data.copy_(torch_weights.view_as(torch_bn.weight.data))
	torch_bn.running_mean.copy_(torch_running_mean.view_as(torch_bn.running_mean.data))
	torch_bn.running_var.copy_(torch_running_var.view_as(torch_bn.running_var.data))

	my_bn.load_bias(my_bias)
	my_bn.load_weights(my_weights)
	my_bn.load_running_mean(my_running_mean)
	my_bn.load_running_var(my_running_var)

	my_blk.add_model(my_bn)
	torch_blk.add_module("batch_norm_{0}".format(0), torch_bn)

	my_lrelu = LRelu(0.1)
	torch_lrelu = nn.LeakyReLU(0.1, inplace = True)

	my_blk.add_model(my_lrelu)
	torch_blk.add_module("leaky_{0}".format(0), torch_lrelu)

	my_result = my_blk.forward(my_inp)
	torch_result = torch_blk(torch_inp)
	
	if check_conv(torch_result, my_result) and check_conv(torch_inp, my_inp):
		print("test conv blk 3 passed")

def test_upsample_1():
	my_inp = get_input()
	torch_inp = get_test_input()

	my_up = Upsample(2)
	torch_up = nn.Upsample(scale_factor = 2, mode = "nearest")

	my_result = my_up.forward(my_inp)
	torch_result = torch_up(torch_inp)

	if check_conv(torch_result, my_result):
		print("test upsample 1 passed")

def test_upsample_2():
	my_inp = get_input()
	torch_inp = get_test_input()

	my_up = Upsample(5)
	torch_up = nn.Upsample(scale_factor = 5, mode = "nearest")

	my_result = my_up.forward(my_inp)
	torch_result = torch_up(torch_inp)

	if check_conv(torch_result, my_result):
		print("test upsample 2 passed")

if __name__ == "__main__":
	'''print(myModule_CUDA.Matmul_Par_CUDA(np.array([[1,2],[3,4]], dtype="float32"), np.array([[1,1],[4,0]], dtype="float32"), np.array([1,1], dtype="float32"), 2, 2, 2, 2))'''
	test_conv_c_par_cuda()
	'''
	test_conv_c_par_omp()
	test_conv_2_c_par_cuda()
	test_conv_3_c_par_cuda()
	test_conv_4_c_par_cuda()
	test_conv_1_c_seq()
	test_conv_2_c_seq()
	test_conv_3_c_seq()
	test_conv_4_c_seq()
	test_conv_4_c_seq()
	test_conv_2()
	test_conv_3()
	test_conv_4()
	test_bn_1()
	test_bn_2()
	test_bn_3()
	test_lrelu_1()
	test_lrelu_2()
	test_conv_blk_1()
	test_conv_blk_2()
	test_conv_blk_3()
	test_upsample_1()
	test_upsample_2()'''