from __future__ import division
import numpy as np
from model import *
import cv2 

def get_input():
	img = cv2.imread("image/eagle.jpg")
	img = cv2.resize(img, (416,416))          #Resize to the input dimension
	img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
	img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
	return img_

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
	inp = get_input()
	my_out = DN.forward(inp)

