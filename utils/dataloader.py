import cv2
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

from utils.utils import cvtColor, preprocess_input


class DepthandPctDataset(Dataset):
    def __init__(self, depth_annotation_path,pct_path, input_shape, num_classes, train, trainmode):
        super(DepthandPctDataset, self).__init__()
        self.depth_annotation_path         = depth_annotation_path
        self.pct_path                      = pct_path
        self.input_shape                   = input_shape
        self.num_classes                   = num_classes
        self.length                        = len(self.depth_annotation_path)
        self.train                         = train
        self.trainmode                     = trainmode

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length

        depth, pct, box  = self.get_random_data(self.depth_annotation_path[index],self.pct_path[index], self.input_shape[0:2], random = self.train)

        depth       = np.transpose(preprocess_input(np.array(depth, dtype=np.float32)), (2, 0, 1))
        pct         = np.transpose(preprocess_input(np.array(pct, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)
        

        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0] 

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return (depth, pct, box)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, depth_annotation_path, pct_path,input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = depth_annotation_path.split()
        line2   = pct_path.split()

        depth   = Image.open("E:\mmYodar\yadar"+line[0][14:].replace("\\", '/'))     # 修改文件路径，每换一次环境，换一下路径
        pct   = Image.open("E:\mmYodar\yadar"+line2[0][14:].replace("\\", '/'))
        

        depth   = cvtColor(depth)
        pct     = cvtColor(pct)  

        iw, ih  = depth.size
        h, w    = input_shape

        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        
        if 1:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2


            depth       = depth.resize((nw,nh), Image.BICUBIC)
            pct         = pct.resize((nw,nh), Image.BICUBIC)         
            new_depth   = Image.new('RGB', (w,h), (128,128,128))
            new_depth.paste(depth, (dx, dy))
            depth_data  = np.array(new_depth, np.float32)
            new_pct   = Image.new('RGB', (w,h), (128,128,128))
            new_pct.paste(pct, (dx, dy))
            pct_data  = np.array(new_pct, np.float32)


            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return depth_data, pct_data, box
        
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    x1s = []
    x2s = []
    bboxes = []
    for x1, x2, box in batch:
        x1s.append(x1)
        x2s.append(x2)
        bboxes.append(box)
    x1s = torch.from_numpy(np.array(x1s)).type(torch.FloatTensor)
    x2s = torch.from_numpy(np.array(x2s)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return x1s, x2s, bboxes