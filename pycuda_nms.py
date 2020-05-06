from __future__ import division

import torch 
import numpy as np
import cv2 
from pycuda.compiler import SourceModule
import pycuda.autoinit
import pycuda.driver as drv
import PIL.Image as Image
import torch
import torchvision
import string
import time
import copy
import math

def cuda_nms(modules, boxes, scores, thresh):
   
    print(boxes)
    num_boxes = boxes.shape[0]
    result = np.array([True]*num_boxes, dtype=np.bool)
    
    # Perform nms on GPU
    NMS_GPU = modules.get_function("NMS_GPU")
    #use drv.InOut instead of drv.Out so the value of results can be passed in
    
    #Setting1:works only when count<=1024
    grid_size, block_size = (1,num_boxes,1), (num_boxes,1,1)
    
    #Setting2:works when count>1024
    #grid_size, block_size = (count,count,1), (1,1,1)
    
    #Setting3:works when count>1024, faster then Setting2
    #block_len = 32
    #grid_len = math.ceil(num_boxes/block_len)
    #grid_size, block_size = (grid_len,grid_len,1), (block_len,block_len,1)
    
    NMS_GPU(drv.In(boxes), drv.InOut(result),
            grid=grid_size, block=block_size)
    print(result)
    return list(np.where(result)[0])

def unique(image_pred_):
    unique_np = np.unique(image_pred_)
    #unique_tensor = torch.from_numpy(unique_np)
    
    #tensor_res = tensor.new(unique_tensor.shape)
    #tensor_res.copy_(unique_tensor)
    return unique_np


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  np.maximum(b1_x1, b2_x1)
    inter_rect_y1 =  np.maximum(b1_y1, b2_y1)
    inter_rect_x2 =  np.minimum(b1_x2, b2_x2)
    inter_rect_y2 =  np.minimum(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1,a_min=0,a_max = None) * np.clip(inter_rect_y2 - inter_rect_y1 + 1,a_min= 0,a_max = None)
    
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    print(iou)
    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):

    
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    
    return prediction

def write_results(prediction, confidence, num_classes, cuda_code,nms_conf = 0.2):
    #confidence == 0.5
    #num_classes = 80
    #prediction = prediction.numpy() 
    #print(type(prediction[0,1,0])) 
    #tmp =(prediction[:,:,4] > confidence).float().unsqueeze(2)
    #print((prediction[:,:,4] > confidence).shape)
    #shape threshhold from [1,10647] => [1,10647,1]
    #print(tmp.shape)
    threshhold = (prediction[:,:,4] > confidence)
    #print(threshhold.shape)
    conf_mask = np.expand_dims(threshhold,axis=2)
    #print(conf_mask.shape)
    prediction = prediction*conf_mask
    #print(prediction)
    box_corner = np.empty_like(prediction)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    print(type(prediction))
    batch_size = prediction.shape[0]

    write = False
    THETA = nms_conf
    


    for ind in range(batch_size):
        image_pred = prediction[ind]          #image Tensor
        
        arr = image_pred[:,5:5+num_classes]
        #max_conf, max_conf_score = image_pred[:,5:5+num_classes].max(axis= 1)
        max_conf = np.max(arr,axis = 1)
        max_conf_score = np.argmax(arr, axis= 1)
        #print(max_conf)
        #print(max_conf_score)
        #print(max_conf.shape)
        #print(max_conf_score.shape)
        #break
        max_conf = np.expand_dims(max_conf,axis=1)
        max_conf_score = np.expand_dims(max_conf_score,axis=1)
        max_conf_score = np.asarray(max_conf_score, dtype=np.float32)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        #print(max_conf)
        #print(max_conf_score)
        #print(max_conf.shape)
        #print(max_conf_score.shape)
        #break
        image_pred = np.concatenate(seq, 1)
        #print(image_pred.shape)
        non_zero_ind =  np.nonzero(image_pred[:,4])
        non_zero_ind = non_zero_ind[0]
        #print(non_zero_ind)
        image_pred_ = image_pred[np.squeeze(non_zero_ind),:]
        try:
            image_pred_ = image_pred[np.squeeze(non_zero_ind),:]
            #print(image_pred_)
        except:
            continue
        #print(image_pred_.shape)
        if image_pred_.shape[0] == 0:
            continue       
#        
  
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index
        #print(img_classes)
        
        for cls in img_classes:
            #perform NMS

            
            #get the detections with one particular class
            image_pred_reshaped = np.expand_dims(image_pred_[:,-1] == cls, axis = 1)
            cls_mask = image_pred_*image_pred_reshaped
            class_mask_ind = np.nonzero(cls_mask[:,-2])
            class_mask_ind = np.squeeze(class_mask_ind)
            image_pred_class = image_pred_[class_mask_ind]
            print(image_pred_class)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
           
            conf_sort_index = np.sort(image_pred_class[:,4])[::-1]
            boxes =  image_pred_class[:,0:4].copy()
            conf_scores = image_pred_class[:,4].copy()
            #print(conf_sort_index)
            cuda_code = string.Template(cuda_code)
            cuda_code = cuda_code.substitute(THETA=THETA)
            modules = SourceModule(cuda_code) 
            # python function will change array's value, so use .copy()
            cuda_start = time.time()
            cuda_results = cuda_nms(modules, boxes, conf_scores, nms_conf)
            cuda_end = time.time()
            print("CUDA results:", cuda_results)
            print("CUDA version takes {} seconds".format(cuda_end-cuda_start))
            print(cuda_results)
            index = cuda_results[0]
            image_pred_class = np.expand_dims(image_pred_class[index], axis = 0)
            print(image_pred_class)
            batch_ind = np.zeros((image_pred_class.shape[0], 1))
            #print(tmp)
            #batch_ind = tmp.fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            
            if not write:
                
                output = np.concatenate(seq,1)
                write = True
            else:
                out = np.concatenate(seq,1)
                output = np.concatenate([output,out])
    try:
        return output
    except:
        return 0
    

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

def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes 
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names