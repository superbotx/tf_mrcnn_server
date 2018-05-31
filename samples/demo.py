import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, request, Response
import jsonpickle
import cv2
import skimage.io
from PIL import Image
import base64
import json
import pickle

ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

app = Flask(__name__)

file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
# results = model.detect([image], verbose=1)

@app.route('/api/detect', methods=['POST'])
def detect():
    global model, image
    r = request
    print ("r", r)
    print ("r.values", r.values)
    # print ("r.data", r.data)
    print("r.data length: ", len(r.data))

    # byte_image = r.data.encode('utf-8')
    decoded_image = base64.decodebytes(r.data)
    print("decoded image size: ", len(decoded_image))
    nparr = np.fromstring(decoded_image, np.uint8)

    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imshow("image", cv2_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # nparr = np.fromstring(r.data, np.uint8)
    # cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_arr = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_arr)
    img.save("tmp.jpg")
    img_sk = skimage.io.imread('tmp.jpg')
    img_np = np.array(img)
    response = {
        'message': 'image received. size={}x{}'.format(cv2_img.shape[1], cv2_img.shape[0])
    }
    print(response)
    print('their image: ', type(image), image.shape)
    print('my image: ', type(img_np), img_np.shape)
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    results = model.detect([img_np], verbose=1)
    r = results[0]
    print(r)
    print(type(r))
    # r = json.dumps(r)
    # r = pickle.dumps(r)
    # r2 = pickle.loads(r)
    r = jsonpickle.encode(r, unpicklable=False)
    r2 = json.loads(r)
    print(r2)
    # r2 = jsonpickle.decode(r2)
    # print(r2)

    # r2 = jsonpickle.decode(r)
    # if r == r2:
    #   print("Decoded successfully.")
    # else:
    #   print("Decoding error.")
    return Response(response=r, status=200, mimetype="application/json")

if __name__ == '__main__':
    app.run(host="localhost", port=5000)
