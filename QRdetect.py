import argparse
import time
from pyzbar import pyzbar
from pathlib import Path
from IPython.display import Image, clear_output
import cv2
import numpy
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image as image
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
class QRdetect:
    def __init__(self, det,im0):
        contexts=[]
        boxes=[]
        #det[:, :4] = scale_coords(im0.shape[2:], det[:, :4], im0.shape).round()
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
           # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        #Write results
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            #label = None if False else (names[c] if True else f'{names[c]} {conf:.2f}')
            d=0
            #plot_one_box(xyxy, im0, color=(255,0,0), line_thickness=1)
            #frame=im0
            height, width = im0.shape[:2]
            x1=int(xyxy[0].item())
            y1=int(xyxy[1].item())
            x2=int(xyxy[2].item())
            y2=int(xyxy[3].item())
            boxes.append([x1,y1,x2,y2])
            if x1-10>=0:
                x11=x1-10
            else:
                x11=0
            if y1-10>=0:
                y11=y1-10
            else:
                y11=0
            if x2+10<=width:
                x22=x2+10
            else:
                x22=width
            if y2+10<=height:
                y22=y2+10
            else:
                y22=height
            if (x11>=0 and y11>=0 and y22<=height and x22<=width):
                img1=im0 [y11:y22,x11:x22]
                #acl = image.fromarray(acl)
                acl=img1 = cv2.cvtColor(numpy.array(img1),cv2.COLOR_BGR2GRAY)
                acl= cv2.resize(acl, (300 , 300))
                if pyzbar.decode(acl):
                    barcodes = pyzbar.decode(acl)
                    for barcode in barcodes:
                        barcodeData = barcode.data.decode("utf-8")
                        barcodeType = barcode.type
                        text = "{} ({})".format(barcodeData, barcodeType)
                        contexts.append(text)
                        d=1
                if d==0:
                    contexts.append('Unknown')
        self.contexts=contexts
        self.boxes=boxes
def CropImg(im0,x1,x2,y1,y2):
    height, width = im0.shape[:2]
    if x1-60>=0:
        x11=x1-60
    else:
        x11=0
    if y1-60>=0:
        y11=y1-60
    else:
        y11=0
    if x2+60<=width:
        x22=x2+60
    else:
        x22=width
    if y2+60<=height:
        y22=y2+60
    else:
        y22=height      
    if (x11>=0 and y11>=0 and y22<=height and x22<=width):
        return x11,x22,y11,y22

class decode:
    def __init__(self,img):
        sta=0  
        acl=img1 = cv2.cvtColor(numpy.array(img1),cv2.COLOR_BGR2GRAY)
        acl= cv2.resize(acl, (300 , 300))
        if pyzbar.decode(acl):
            barcodes = pyzbar.decode(acl)
        for barcode in barcodes:
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            text = "{} ({})".format(barcodeData, barcodeType)
            self.contexts.append(text)
            sta==1
        if sta==0:
            self.contexts.append('Unknown')