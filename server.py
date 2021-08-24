import cv2
import json
import base64
import numpy as np
# import request
from flask      import Flask, request
from flask_cors import CORS, cross_origin
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


def decode_img(img_base64):
    img = img_base64.encode()
    img = base64.b64decode(img)
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, flags=cv2.IMREAD_COLOR)
    return img


# Logging setup
logging.setup()

app = Flask(__name__)
cors = CORS(app, resources={r"/face/*": {"origins": "*"}})
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
face_id = FaceID()
conn = face_id.connect_database
mode=None

@app.route('/mode', methods=['POST'])
@cross_origin(supports_credentials=True)
def set_mode():
    try:
        request_data = request.get_json(force=True)
        global mode
        mode = request_data.get('mode')
        logging.debug("Get Mode: {}".format(mode))
        return json.dumps({'success':True}), 200, {'ContentType':'application/json'}
    except Exception as e:
        logging.error('face_recognize have error: {}'.format(str(e)))
        return "Internal Server Error", 500


@app.route('/mode', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_mode():
    try:
        global mode
        response_data = {
            "mode": mode
        }
        return response_data, 200
    except Exception as e:
        logging.error('face_recognize have error: {}'.format(str(e)))
        return "Internal Server Error", 500

@app.route('/face/recognize', methods=['POST'])
@cross_origin(supports_credentials=True)
def face_recognize():
    try:
        global mode
        request_data = None
        logging.debug("Face Recognize: {}".format(mode))
        if mode != "register":
            request_data = request.get_json(force=True)
            faces = request_data.get('user_faces')

            person_names = []
            badge_ids = []
            distances=[]
            if len(faces) != 0:
                for face in faces:
                    # Decode Image
                    face_img = decode_img(face)
                    logging.info("*************Decoded Image*************")

                    # Preprocess
                    # TODO
                    user_id, badge_id, distance = face_id.get_face(face_img)
                    # Recognize
                    person_names.append(user_id)
                    badge_ids.append(badge_id)
                    distances.append(distance)
                    logging.info("*************Recognized Face*************")
                    logging.info("____________________________________________________________")

                # Generate response data
                response_data = {
                    "user_ids": person_names,
                    "badge_ids": badge_ids,
                    "distances":distances
                }
                logging.info('Response Data: {}'.format(str(response_data)))

            return response_data, 200
        
        response_data = {
            "mode": mode
        }
        return response_data, 200
        
    except Exception as e:
        logging.error('face_recognize have error: {}'.format(str(e)))
        return "Internal Server Error", 500


@app.route('/face/register', methods=['POST'])
@cross_origin(supports_credentials=True)
def face_register():
    try:
        global mode
        # Request Format
        # {
        # "user_faces": ["base_64(face[0])","base_64(face[1])","base_64(face[2])",\
        #               "base_64(face[3])","base_64(face[4])","base_64(face[5])"],
        # "user_id": "<user_id>",
        # "camera_id": "camera_id"
        # }
        
        request_input = request.get_json(force=True)
        user_faces = request_input.get('user_faces')
        user_id = request_input.get('user_id')
        
        mode = "register"

        # Simple validate
        if user_faces is None or user_id is None:
            response_data = {
                'success':False,
                'message': "Input faces or user id is none"
                }
            return json.dumps(response_data), 400, {'ContentType':'application/json'}

        # Preprocess
        input_images = []

        for img in user_faces:
            # Decode Image
            face_img = decode_img(img)

            # Add to process list
            input_images.append(face_img)

        # Store face in database
        for idx, face_img in enumerate(input_images):
            # Register new face and generate face with facemask
            # stat = True <=> successful
            is_success, status = face_id.update_face(
                image=face_img,
                idx=idx,
                badge_id="abc123",
                img_url="img_url",
                name='{}'.format(user_id))

            # if register or update fail
            if is_success == False:
                response_data = {
                    'success': is_success,
                    'status': status,
                    'message': "{} fail".format(status)
                }
                return json.dumps(response_data), 400, {'ContentType':'application/json'}

        logging.info('Successful face registration!')
        response_data = {
            'success': is_success,
            'status': status,
            'message': "{} success".format(status)
        }
        mode = None
        return json.dumps(response_data), 200, {'ContentType':'application/json'}
    except Exception as e:
        mode = None
        logging.error('face_register have error: {}'.format(str(e)))
        return "Internal Server Error", 500

@app.route('/face/register/sync', methods=['POST'])
@cross_origin(supports_credentials=True)
def face_register_sync():
    try:
        global mode
        # Request Format
        # {
        # "user_faces": ["base_64(face[0])","base_64(face[1])","base_64(face[2])",\
        #               "base_64(face[3])","base_64(face[4])","base_64(face[5])"],
        # "user_id": "<user_id>",
        # "camera_id": "camera_id"
        # }
        
        request_input = request.get_json(force=True)
        request_input = json.loads(request_input)

        img_urls = request_input.get('picURI')
        badge_id = request_input.get('customId')
        user_id = request_input.get('name')
        
        mode = "register"
        # Simple validate
        if img_urls is None or user_id is None:
            response_data = {
                'success':False,
                'message': "Input faces or user id is none"
                }
            logging.info('Fail to register ( invalid input face or none id )') 
            return json.dumps(response_data), 400, {'ContentType':'application/json'}

        # Store face in database
        for idx, img_url in enumerate(img_urls):
            # Register new face and generate face with facemask
            # stat = True <=> successful
            is_success, status = face_id.update_face(
                image=helps.read_image_url(img_url),
                idx=idx,
                badge_id=badge_id,
                img_url=img_url,
                name='{}'.format(user_id))

            # if register or update fail
            if is_success == False:
                response_data = {
                    'success': is_success,
                    'status': status,
                    'badge_id': badge_id, 
                    'message': "{} fail".format(status)
                }
                logging.info('Fail to register ( register or update fail )')
                return json.dumps(response_data), 400, {'ContentType':'application/json'}

        logging.info('Successful face registration!')
        response_data = {
            'success': is_success,
            'status': status,
            'badge_id': badge_id,
            'message': "{} success".format(status)
        }
        mode = None
        return json.dumps(response_data), 200, {'ContentType':'application/json'}
    except Exception as e:
        mode = None
        logging.error('face_register have error: {}'.format(str(e)))
        return "Internal Server Error", 500


@app.route('/face/<badge_id>', methods=['DELETE'])
@cross_origin(supports_credentials=True)
def face_remove(badge_id):
    try:
        # Simple validate
        if badge_id is None:
            response_data = {
                'success': False,
                'message': "Badge ID is None"
            }
            return json.dumps(response_data), 400, {'ContentType':'application/json'}
        
        # stat = True <=> successful
        stat = face_id.remove_face(badge_id)

        if stat == False:
            response_data = {
                'success': False,
                'badge_id': badge_id,
                'message': "The user id does not exist in the database"
            }
            return json.dumps(response_data), 400, {'ContentType':'application/json'}
        else:
            response_data = {
                'success': True,
                'badge_id': badge_id,
                'message': "Face removal successfully"
            }
            logging.info('Face removal successfully!')
            return json.dumps(response_data), 200, {'ContentType':'application/json'}

    except Exception as e:
        logging.error('face_remove have error {}'.format(str(e)))
        return "Internal Server Error", 500


@app.route('/face/search', methods=['POST'])
@cross_origin(supports_credentials=True)
def search_face():
    try:
        request_input = request.get_json(force=True)
        user_id = request_input.get('user_id')
        rows = face_database.search_by_name(conn, value=user_id)
        return json.dumps(rows), 200, {'ContentType':'application/json'}
    except Exception as e:
        logging.error('face_remove have error {}'.format(str(e)))
        return "Internal Server Error", 500

@app.route('/face/get_all', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def get_all():
    try:
        request_input = request.get_json(force=True)
        rows = face_database.get_all(conn)
        return json.dumps(rows), 200, {'ContentType':'application/json'}
    except Exception as e:
        logging.error('get all have error {}'.format(str(e)))
        return "Internal Server Error", 500



if __name__ == '__main__':
    # Start server
    app.run(debug=False, host='0.0.0.0', port=5555)
