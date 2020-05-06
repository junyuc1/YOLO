from __future__ import division
import numpy as np
from model import *
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import cv2 
import time
import myModule

def get_input(name):
	img = cv2.imread(name)
	img = cv2.resize(img, (416,416))          #Resize to the input dimension
	img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
	img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
	return img_.astype(np.float32)

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
	"""
	Prepare image for inputting to the neural network. 

	Returns a Variable 
	"""
	img = (letterbox_image(img, (inp_dim, inp_dim)))
	img = img[:,:,::-1].transpose((2,0,1)).copy()
	img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
	return img

def parse_cfg(cfgfile):
	"""
	Takes a configuration file

	Returns a list of blocks. Each blocks describes a block in the neural
	network to be built. Block is represented as a dictionary in the list

	"""

	file = open(cfgfile, 'r')
	lines = file.read().split('\n')                        # store the lines in a list
	lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
	lines = [x for x in lines if x[0] != '#']              # get rid of comments
	lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces

	block = {}
	blocks = []

	for line in lines:
		if line[0] == "[":               # This marks the start of a new block
			if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
				blocks.append(block)     # add it the blocks list
				block = {}               # re-init the block
			block["type"] = line[1:-1].rstrip()     
		else:
			key,value = line.split("=") 
			block[key.rstrip()] = value.lstrip()
	blocks.append(block)

	return blocks

def get_network(model_cfg, model_weights):   
	network = []
	depth = 3
	filter_list = []
	weights = open(model_weights, "rb")
	
	weights = np.fromfile(weights, dtype = np.float32)[5:]
	weight_pos = 0
	for i in range(len(model_cfg)):
		cfg = model_cfg[i]
		model_type = cfg["type"]

		if (model_type == "convolutional"):
			conv_blk = Conv_Block()

			num_filters = int(cfg["filters"])
			size = int(cfg["size"])
			stride = int(cfg["stride"])
			pad = int(cfg["pad"])
			activation = cfg["activation"]
			pad = (size - 1) // 2 if pad == 1 else 0

			conv = Convolution(depth, num_filters, size, stride, pad)
			conv_blk.add_model(conv)

			if (activation == "leaky"):
				bn = Batch_Normalize(num_filters)
				conv_blk.add_model(bn)

				lrelu = LRelu(0.1)
				conv_blk.add_model(lrelu)

				conv_blk.models[1].load_bias(weights[weight_pos : weight_pos + num_filters])
				conv_blk.models[1].load_weights(weights[weight_pos + num_filters : weight_pos + 2 * num_filters])
				conv_blk.models[1].load_running_mean(weights[weight_pos + 2 * num_filters : weight_pos + 3 * num_filters])
				conv_blk.models[1].load_running_var(weights[weight_pos + 3 * num_filters : weight_pos + 4 * num_filters])

				weight_pos += 4 * num_filters
			else:
				conv_blk.models[0].load_bias(weights[weight_pos : weight_pos + num_filters])

				weight_pos += num_filters

			conv_blk.models[0].load_weights(weights[weight_pos : weight_pos + num_filters * size * size * depth])
			weight_pos += num_filters * size * size * depth

			depth = num_filters
			filter_list.append(depth)
			network.append(conv_blk)

		elif (model_type == "upsample"):
			up = Upsample(int(cfg["stride"]))
			depth = depth
			filter_list.append(depth)
			network.append(up)

		elif (model_type == "route"):
			layers = cfg["layers"].split(',')
			cfg["layers"] = cfg["layers"].split(',')
	
			start = int(layers[0])
			end = 0 if len(layers) == 1 else int(layers[1])

			start_delta = start - i if start > 0 else start
			end_delta = end - i if end > 0 else end
			
			if len(layers) == 2:
				depth = filter_list[i + start_delta] + filter_list[i + end_delta]
				rt = Route([i + start_delta, i + end_delta])
			else:
				depth = filter_list[i + start_delta]
				rt = Route([i + start_delta])

			filter_list.append(depth)
			network.append(rt)

		elif (model_type == "shortcut"):
			from_pos = int(cfg["from"])
			st = Shortcut(i - 1, i + from_pos)
			depth = depth
			filter_list.append(depth)
			network.append(st)

		elif (model_type == "yolo"):
			mask = cfg["mask"].split(",")
			anchors = cfg["anchors"]
			anchors = anchors.split(",")
			this_anchor = [(int(anchors[int(m) * 2]), int(anchors[int(m) * 2 + 1])) for m in mask]
			num_class = int(cfg["classes"])
			yl = Yolo(this_anchor, num_class)

			depth = depth
			filter_list.append(depth)
			network.append(yl)

	return network

class DarkNet:
	def __init__(self):
		self.model_cfg = parse_cfg("cfg/yolov3.cfg")
		model_weights = "yolov3.weights"
		hyper_params = self.model_cfg[0]
		self.image_size = hyper_params["width"]
		self.model_cfg = self.model_cfg[1:]
		self.network = get_network(self.model_cfg, model_weights)

	def forward(self, input):
		output = input
		layers_out = []
		first_yolo = True
		pred = None

		for i in range(len(self.model_cfg)):
			cfg = self.model_cfg[i]
			model_type = cfg["type"]
			cur_model = self.network[i]

			if (model_type == "convolutional" or model_type == "upsample"):
				output = cur_model.forward(output)
				layers_out.append(output)

			elif (model_type == "route" or model_type == "shortcut"):
				output = cur_model.forward(layers_out)
				layers_out.append(output)
			
			elif (model_type == "yolo"):
				output = cur_model.forward(self.image_size, output)
				layers_out.append(output)
				if first_yolo:
					pred = output
					first_yolo = False
				else:
					pred = np.concatenate((pred, output), axis=1)

		return pred

if __name__ == "__main__":
	DN = DarkNet()
	imp = []
	for i in range(1):
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
		imp.append(im1)
		imp.append(im2)
		imp.append(im3)
		imp.append(im4)
		imp.append(im5)
		imp.append(im6)
		imp.append(im7)
		imp.append(im8)

	start = time.time()
	imp = np.array(imp)
	out = DN.forward(imp)

	



	
	end = time.time()
	print("Time taken: " + str(end - start))


	

