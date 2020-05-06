import numpy as np
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



template = """
#define THETA $THETA
#include <stdio.h>
__device__
float IOUcalc(float* b1, float* b2)
{
    //y1, x1, h, w
    float ai = (float)b1[3]*b1[2];
    float aj = (float)b2[3]*b2[2];
    float x_inter, x2_inter, y_inter, y2_inter;
    x_inter = max(b1[1],b2[1]);
    y_inter = max(b1[0],b2[0]);
    x2_inter = min((b1[1] + b1[3]),(b2[1] + b2[3]));
    y2_inter = min((b1[0] + b1[2]),(b2[0] + b2[2]));
    
    float w = (float)max((float)0, x2_inter - x_inter);  
    float h = (float)max((float)0, y2_inter - y_inter);  
    float inter = ((w*h)/(ai + aj - w*h));
    return inter;
}
__global__
void NMS_GPU(float *d_b, bool *d_res)
{
    int abs_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int abs_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if(d_b[abs_x*5+4] < d_b[abs_y*5+4])
    {
        float* b1 = &d_b[abs_y*5];
        float* b2 = &d_b[abs_x*5];
                
        float iou = IOUcalc(b1,b2);
                
        if(iou>THETA)
        {
            d_res[abs_x] = false; 
        }
    }
}
"""

def to_yxhw(boxes):
    # box format: [y1,x1,y2,x2] to [y1,x1,h,w]
    boxes[:,2:] = boxes[:,2:]-boxes[:,0:2]
    return boxes

def cuda_nms(modules, boxes, scores, yxhw=False):
    if not yxhw:
        boxes = to_yxhw(boxes)
    
    n_boxes  = boxes.shape[0]
    boxes = np.hstack((boxes, np.expand_dims(scores,axis=1)))
    results = np.array([True]*n_boxes, dtype=np.bool)
    
    # Perform nms on GPU
    NMS_GPU = modules.get_function("NMS_GPU")
    #use drv.InOut instead of drv.Out so the value of results can be passed in
    
    #Setting1:works only when count<=1024
    #grid_size, block_size = (1,n_boxes,1), (n_boxes,1,1)
    
    #Setting3:works when count>1024, faster then Setting2
    thread_per_block_dim = 32
    grid_len = math.ceil(n_boxes/thread_per_block_dim)
    grid_size, block_size = (grid_len,grid_len,1), (thread_per_block_dim,thread_per_block_dim,1)
    
    NMS_GPU(drv.In(boxes), drv.InOut(results),
            grid=grid_size, block=block_size)
    return list(np.where(results)[0])

def python_naive_nms(boxes, thresh):
    #from Felzenszwalb et al
    pick = []

    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)


            iou = float(w * h) / area[j]


            if iou > thresh:
                suppress.append(pos)

       
        idxs = np.delete(idxs, suppress)

    return boxes[pick]

if __name__ == "__main__":
    threshhold=0.4
    
    boxes = np.loadtxt('boxes.txt', dtype=np.float32)
    scores = np.loadtxt('scores.txt', dtype=np.float32)
    num_boxes = 5000
    boxes_try = boxes[0:num_boxes,]
    scores_try = scores[0:num_boxes,]
    template = string.Template(template)
    template = template.substitute(THETA=threshhold)
    modules = SourceModule(template) 
    # python function will change array's value, so use .copy()
    cuda_start = time.time()
    cuda_results = cuda_nms(modules, boxes_try.copy(), scores_try.copy())
    cuda_end = time.time()
    #print("CUDA results:", cuda_results)
    print("CUDA version takes {} seconds".format(cuda_end-cuda_start))


    tb = boxes_try.copy()
    s = scores_try.copy()
    torch_boxes = torch.from_numpy(tb)
    torch_scores = torch.from_numpy(s)
    torch_start = time.time()
    torch_res = torchvision.ops.nms(torch_boxes, torch_scores, 0.4)
    torch_end = time.time()
    #print("Torch results:",torch_res)
    print("Torch version takes {} seconds".format(torch_end-torch_start))

    py_start = time.time()
    python_res = python_naive_nms(boxes_try.copy(), 0.4)
    py_end = time.time()
    #print("Python results:", python_res)
    print("Python version takes {} seconds".format(py_end-py_start))

    

