import argparse
import time
from selenium.webdriver.chrome.options import Options  # for suppressing the browser
from numpy.lib.ufunclike import fix
from pyzbar import pyzbar
from pathlib import Path
from IPython.display import Image, clear_output
import cv2
import numpy
import os
import glob
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
from QRdetect import QRdetect,CropImg
import time
import cv2 
from selenium import webdriver
  
# Here Chrome  will be used
option = webdriver.ChromeOptions()
option.add_argument('headless')
driver = webdriver.Chrome(options=option)
from datetime import datetime
from flask import Flask, render_template, Response, request
from detect1 import mode,Load
from flask_socketio import SocketIO, emit
async_mode = None
app = Flask(__name__,template_folder='templates')
socketio = SocketIO()
socketio.init_app(app, async_mode="threading")

@socketio.on('connect')
def test_connect():
    print("client connected")
    emit('message',  {'data':'Lets dance'})

@socketio.on('Slider value changed')
def value_changed(message):
    socketio.emit('update value', message, broadcast=True)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('1index1.html', sync_mode=socketio.async_mode)

def gen():
    """Video streaming generator function."""
    a=Load()
    dt = datetime.now()
    today=strg = dt.strftime("%b%d%Y%H%M%S")
    #print(strg)  # July 22, 2017
    model,device,imgsz=a.macdinh()
    #view_img = check_imshow()
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    d=0
    cap= cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('rtspsrc location=rtsp://admin:12345678x@X@192.168.129.241 rtph264depay ! h264parse ! omxh264dec ! nvvidconv !video/x-raw, width=(int)1280, height=(int)720,!format=(string)BGRx !  videorate ! video/x-raw-yuv,framerate=25/2 !queue ! videoconvert ! appsink',cv2.CAP_GSTREAMER)
    #v4l2src device=rtsp://admin:12345678x@X@192.168.129.124/ io-mode=2 ! image/jpeg, width=(int)1280, height=(int)720 !  jpegdec ! video/x-raw!videorate ! video/x-raw-yuv,framerate=25/2 ! videoconvert ! video/x-raw,format=BGR ! appsink')
    dem=0
    poss=0
    filename=[]
    while(cap.isOpened()):
        ret, frame1 = cap.read()
        if dem!=10:   
            dem=dem+1
            frame = cv2.imencode('.jpg', frame1)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else:
            dem=0
            d=d+1
            frame2=frame1.copy()
            img=a.fiximg(frame1)
            t1 = time_synchronized()
            print("detectingggggggggggggggg")
            pred = model(img, augment='store_true')[0]
            # Apply NMS
            pred = non_max_suppression(pred, 0.45, 0.45, agnostic='store_true')
            t2 = time_synchronized()
            im0=frame1.copy()
            #cv2.rectangle(im1, (10,10), (im0.,y2), (0,0,255), 3)
            fileanh=[]
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 siz
                    p=QRdetect(det,im0)
                    for i in range(len(p.boxes)):
                        x1,y1,x2,y2=p.boxes[i]
                        text=p.contexts[i]
                        #print(text)
                        if text=='Unknown':
                            im1=frame1.copy()
                            cv2.rectangle(im1, (x1,y1), (x2,y2), (0,0,255), 3)
                            text_color = (0,0,255)
                            cv2.putText(im1,text, (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2, text_color, thickness=1)
                            filename.append(text)
                            frame = cv2.imencode('.jpg', im1)[1].tobytes()
                            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                            x11,x22,y11,y22=CropImg(im0,x1,x2,y1,y2)
                            img2=frame2[y11:y22,x11:x22]
                            im2 = image.fromarray(img2, 'RGB')
                            img2_path = 'static/'+today+str(poss)+'.png'
                            im2.save(img2_path)
                            poss=poss+1


                        else:
                            im1=frame1.copy()
                            cv2.rectangle(im1, (x1,y1), (x2,y2), (0,220,0), 3)
                            text_color = (0,220,0)
                            cv2.putText(im1,text, (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2, text_color, thickness=1)
                            x11,x22,y11,y22=CropImg(im0,x1,x2,y1,y2)
                            text=text[0:-8]
                            img2=frame2[y11:y22,x11:x22]
                            im2 = image.fromarray(img2, 'RGB')
                            img2_path = 'static/'+today+str(poss)+'.png'
                            im2.save(img2_path)
                            print(text)
                            if text not in filename:
                                print(text)
                                if (('html' in text) or ('http' in text)) and ((text[0]=='h') or (text[0]=='H')):
                                    filename.append(text)
                                    driver.get(text)
                                    get_title = driver.title 
                                else:
                                    filename.append(text)
                                    get_title = text
                                    text='https://www.google.com/'
                                print(filename,'dsds')           
                                emit_message(img2_path, get_title,text)
                                poss=poss+1
                                frame = cv2.imencode('.jpg', im1)[1].tobytes()
                                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def emit_message(img_path,get_title, text):
    with app.app_context():
        socketio.emit('message', {'img_path':img_path ,'name':get_title, "text": text}, namespace="/result")

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
'''
from flask import Flask, render_template, Response
from camera import VideoCamera
app=Flask(__name__,template_folder='templates')


@app.route('/c')
#def index():
 #   return render_template('index.html')
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
'''

if __name__ == '__main__':
    socketio.run(app)
