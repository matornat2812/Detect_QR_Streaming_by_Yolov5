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
from QRdetect import QRdetect
# Load model
from detect1 import mode

import numpy as np
import cv2
model = attempt_load('yolo5s/best.pt')  # load FP32 model

cap = cv2.VideoCapture(0)
    

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()  
    demanh=[]
    model = attempt_load('yolo5s/best.pt')  # load FP32 model
    device= device = select_device('')
    half = device.type != 'cpu' 
    imgsz=640
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams('0', img_size=imgsz, stride=stride)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # Run inference
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment='store_true')[0]
        # Apply NMS
        pred = non_max_suppression(pred, 0.45, 0.45, agnostic='store_true')
        t2 = time_synchronized()
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            p = Path(p)  # to Path
            im1=im0.copy()
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
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
                        demanh.append(im1[y1:y2,x1:x2])
                    else:
                        cv2.rectangle(im1, (x1,y1), (x2,y2), (0,220,0), 3)
                        text_color = (0,220,0)
                        cv2.putText(im1,text, (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2, text_color, thickness=1)
                        demanh.append(im1[y1:y2,x1:x2])
                #for i in demanh:
                # cv2.imshow('manhinh1',demanh[i])
                # cv2.waitkey(2)

            if view_img:
                print(im1.shape[1], im1.shape[0])
                cv2.imshow("manhinh", im1)
                cv2.waitKey(1)  # 1 millisecond
            #confidence_score = conf
            #class_index = cls
            #object_name = names[int(cls)]
            # Stream results
            
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
#xuly()