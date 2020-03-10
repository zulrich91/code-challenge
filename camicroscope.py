from flask import jsonify, Flask, send_file, render_template, redirect, request
import pandas as pd
from flask_cors import CORS, cross_origin
import csv, json
from pathlib import PurePath, Path
import os
import torch
from torchvision import datasets, transforms, models
import torchvision
import numpy as np
from PIL import Image
import cv2
import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
path = Path('./images')
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


#CORS(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["IMAGE_UPLOADS"] = path

#CORS Headers 
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,true')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PATCH,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def get_hello():
    return "Hello world"

# Route to upload image
@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if request.method == "POST":
        if request.files:
            image=request.files["image"]
            print(image)
            print("Name of image is " +image.filename)
            print(path/image.filename)
            image.save(str(path/image.filename))
            full_filename = os.path.join(app.config['IMAGE_UPLOADS'], image.filename)
            detected_file = object_detection_api(full_filename, threshold=0.9)
            return send_file(detected_file,mimetype='image/png')
    return render_template('upload_image.html')


def get_prediction(img_path, threshold):
    img = Image.open(img_path) # Load the image
    transform = torchvision.transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    pred = model([img]) # Pass the image to the model
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]

    return pred_boxes, pred_class

def object_detection_api(img_path, threshold=0.5, rect_th=1, text_size=1, text_th=1):
    boxes, pred_cls = get_prediction(img_path, threshold) # Get predictions
    img = cv2.imread(img_path) # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(231, 237, 36 ), thickness=rect_th) # Draw Rectangle with the coordinates
        cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (231, 237, 36 ),thickness=text_th) # Write the prediction class
        cv2.imwrite(os.path.join(app.config['IMAGE_UPLOADS'],"detected.png"), img)
    return os.path.join(app.config['IMAGE_UPLOADS'],"detected.png")

app.run()