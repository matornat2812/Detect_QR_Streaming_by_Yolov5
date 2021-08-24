import argparse
import time
from numpy.lib.ufunclike import fix
from pyzbar import pyzbar
from pathlib import Path
from IPython.display import Image, clear_output
import cv2
import numpy
from numpy import stack, ascontiguousarray,unique
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image as image
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from QRdetect import QRdetect, CropImg
# Load model
class Load:
    def __init__(self):
        self.model=attempt_load('yolo5s/best.pt')
        self.stride=int(self.model.stride.max())
        self.device= device = select_device('')
        self.half = device.type != 'cpu' 
        self.imgsz=check_img_size(640,s=self.stride)
    def macdinh(self):
        return self.model,self.device,self.imgsz
    def fiximg(self,img):
        s = stack([letterbox(img, self.imgsz, stride=self.stride)[0].shape], 0)  # shapes
        self.rect = unique(s, axis=0).shape[0] == 1
        img = [letterbox(img, self.imgsz, auto=self.rect, stride=self.stride)[0]]
        img = stack(img, 0)
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img
class mode:
    def __init__(self):    
        self.a=Load()
        model,device,imgsz=self.a.macdinh()
        #view_img = check_imshow()
        self.model=model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    def framemodel(self,frame):
        img=self.a.fiximg(frame)
        t1 = time_synchronized()
        pred = self.model(img, augment='store_true')[0]
        # Apply NMS
        pred = non_max_suppression(pred, 0.45, 0.45, agnostic='store_true')
        t2 = time_synchronized()
        im0=frame.copy()
        im1=frame.copy()
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 siz
                p=QRdetect(det,im0)
                for i in range(len(p.boxes)):
                    x1,y1,x2,y2=p.boxes[i]
                    text=p.contexts[i]
                    #print(text)
                    if text=='Unknown':
                        cv2.rectangle(im1, (x1,y1), (x2,y2), (0,0,255), 3)
                        text_color = (0,0,255)
                        cv2.putText(im1,text, (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2, text_color, thickness=1)
                    else:
                        cv2.rectangle(im1, (x1,y1), (x2,y2), (0,220,0), 3)
                        text_color = (0,220,0)
                        cv2.putText(im1,text, (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2, text_color, thickness=1)
        #print(f'{s}Done. ({t2 - t1:.3f}s)')
        return im1
'''cap = cv2.VideoCapture(0)
while(True):
    a=mode()
    ret, frame = cap.read()
    im1=a.framemodel(frame)
    cv2.imshow('frame', im1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()'''
'''def xuly():    
    a=Load()
    model,device,imgsz=a.macdinh()
    #view_img = check_imshow()
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        img=a.fiximg(frame)
        t1 = time_synchronized()
        pred = model(img, augment='store_true')[0]
        # Apply NMS
        pred = non_max_suppression(pred, 0.45, 0.45, agnostic='store_true')
        t2 = time_synchronized()
        for i, det in enumerate(pred):  # detections per image
            im0=frame.copy()
            im1=frame.copy()
            if len(det):
                # Rescale boxes from img_size to im0 siz
                p=QRdetect(det,im0)
                for i in range(len(p.boxes)):
                    x1,y1,x2,y2=p.boxes[i]
                    text=p.contexts[i]
                    #print(text)
                    if text=='Unknown':
                        cv2.rectangle(im1, (x1,y1), (x2,y2), (0,0,255), 3)
                        text_color = (0,0,255)
                        cv2.putText(im1,text, (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2, text_color, thickness=1)
                    else:
                        cv2.rectangle(im1, (x1,y1), (x2,y2), (0,220,0), 3)
                        text_color = (0,220,0)
                        cv2.putText(im1,text, (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2, text_color, thickness=1)
        #print(f'{s}Done. ({t2 - t1:.3f}s)')


            cv2.imshow('frame', im1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()'''